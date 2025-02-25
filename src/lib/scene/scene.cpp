#include "scene/scene.hpp"
#include "model/euler.hpp"
#include "model/multi.hpp"
#include "model/navier_stokes.hpp"
#include "object_mesh.hpp"
#include <igl/opengl/glfw/Viewer.h>

namespace dacti::scene {

	template<typename M>
	integrator::Boundary<M> Scene<M>::nearest_boundary(vecn_t point) const {
		scalar_t min_dist = std::numeric_limits<scalar_t>::max();
		integrator::Boundary<M> bc_min;

		for (int mi = 0; mi < config->objects.size(); mi++) {
			if (!config->objects[mi].bc) continue;

			vecn_t c, n;

			double dist = nearest_mesh_point(config->objects[mi].mesh, point, c, n);

			if (dist < min_dist) {
				assert(config->objects[mi].bc);
				min_dist = dist;
				bc_min   = config->objects[mi].bc.value()(c, n / n.norm());
			}
		}
		return bc_min;
	}

	template<typename M>
	bool Scene<M>::raycast(vecn_t a, vecn_t b, scalar_t &t) const {
		for (int mi = 0; mi < config->objects.size(); mi++) {
			if (!config->objects[mi].bc) {
				return false;
			}
			if (intersect(config->objects[mi].mesh,
			              a.template project<3>(),
			              b.template project<3>(),
			              t)) {
				return true;
			}
		}
		return false;
	}

	template<typename M>
	void Scene<M>::compute_signed_distances(domain_t &domain, bool set_active) {
		signed_distances.resize(domain.tree.size());
		nearest_mesh_idx.resize(domain.tree.size());

#pragma omp parallel for
		for (int i = 0; i < node_idx.size(); i++) {
			signed_distances[node_idx[i]] = std::numeric_limits<scalar_t>::max();
			nearest_mesh_idx[node_idx[i]] = 0;
		}

		for (int mi = 0; mi < config->objects.size(); mi++) {

			const auto &mesh = config->objects[mi].mesh;

			igl::WindingNumberAABB<Eigen::RowVector3d, Eigen::MatrixXd, Eigen::MatrixXi> wn_aabb(mesh.V, mesh.F);

#pragma omp parallel for
			for (int i = 0; i < node_idx.size(); i++) {
				size_t index = node_idx[i];
				node_t &node = domain.tree[index];
				cell_t &cell = node.cell;

				scalar_t dist = igl::signed_distance_winding_number(mesh.aabb, mesh.V, mesh.F, wn_aabb, cell.center.template to_iterable<Eigen::RowVector3d>());

				if (dist < signed_distances[index]) {
					nearest_mesh_idx[index] = mi;
					signed_distances[index] = dist;
				}

				if (set_active) {
					if (signed_distances[index] < 0.0) {
						node.is_active = !config->objects[nearest_mesh_idx[index]].bc.has_value();
					} else {
						node.is_active = true;
					}
				}
			}
		}
	}


	template<typename M>
	void Scene<M>::compute_node_boundaries(domain_t &domain, std::unordered_set<size_t> &split_set, bool raytrace) {
#pragma omp parallel for
		for (int i = 0; i < node_idx.size(); i++) {
			size_t index = node_idx[i];
			node_t &n    = domain.tree[index];
			cell_t &cell = n.cell;

			bool has_open_face       = false;
			bool has_object_boundary = false;

			n.boundaries.clear();

			for (int dim = 0; dim < M::NumDims; dim++) {
				for (int dir = 0; dir < 2; dir++) {

					vecn_t neighbor_center = cell.center + vecn_t::unit(dim) * (2 * dir - 1) * cell.size;

					if (config->domain_bc[dim] && neighbor_center[dim] * (2 * dir - 1) > config->active_size[dim] / 2.0) {

						n.boundaries.get(dim, dir) = config->domain_bc[dim].value()[dir](cell.center, -vecn_t::unit(dim) * (2 * dir - 1));

					} else {
						size_t neighbor_index = domain.find(neighbor_center, n.tree_level);

						bool signed_distances_boundary = domain.tree[neighbor_index].is_active != n.is_active;

						scalar_t t;
						bool ray_boundary = raycast(cell.center, neighbor_center, t) && raytrace;

						if (signed_distances_boundary || ray_boundary) {
							n.boundaries.get(dim, dir) = nearest_boundary(cell.get_center());
							n.can_unsplit              = false;
							has_object_boundary        = true;

						} else {
							has_open_face = true;
						}
					}
				}
			}

			if (!has_open_face) {
				n.is_active = false;
			}

			if (has_object_boundary) {
				n.can_unsplit = false;
				n.is_boundary = true;
			}

			if (n.should_split) {
#pragma omp critical
				{
					split_set.insert(index);
				}
			}
		}
	}


	template<typename M>
	void Scene<M>::detect_splits(domain_t &domain, std::unordered_set<size_t> &split_set) {
#pragma omp parallel for
		for (int i = 0; i < node_idx.size(); i++) {
			size_t index = node_idx[i];
			node_t &n    = domain.tree[index];
			cell_t &cell = n.cell;

			// split if below the minimum refinement level
			if (n.tree_level < config->min_refinement_level) {
				n.should_split = true;
				n.can_unsplit  = false;
			}

			// split if close to a geometry
			if (std::abs(signed_distances[index]) < cell.size * config->refinement_radius) {
				if (config->objects[nearest_mesh_idx[index]].type == "solid") {
					if (n.tree_level < config->max_refinement_level_geom) {
						n.should_split = true;
						n.can_unsplit  = false;
					}
				} else {
					if (n.tree_level < config->max_refinement_level) {
						n.should_split = true;
						n.can_unsplit  = true;
					}
				}
			}

			// split if close to a inflow boundary
			for (int dim = 0; dim < M::NumDims; dim++) {
				for (int dir = 0; dir < 2; dir++) {
					if (n.boundaries.get(dim, dir)) {
						if (n.boundaries.get(dim, dir)->type == integrator::Boundary<M>::Type::inflow_r) {
							if (n.tree_level < config->max_refinement_level) {
								n.should_split = true;
								n.can_unsplit  = true;
							}
						}
					}
				}
			}

			// split if close to an initial jump
			if (config->ic.type == "jump") {
				if (std::abs(cell.center[config->ic.jump_axis] - config->ic.jump_location) < cell.size) {
					if (n.tree_level < config->max_refinement_level) {
						n.should_split = true;
						n.can_unsplit  = true;
					}
				}
			}

			for (const auto &zone: config->refine_zones) {
				if (zone.contains(cell.center.template project<3>()) && n.tree_level < zone.refinement_level) {
					n.should_split = true;
				}
			}

			if (n.should_split) {
#pragma omp critical
				{
					split_set.insert(index);
				}
			}
		}
	}

	template<typename M>
	void Scene<M>::split_detected_nodes(std::unordered_set<size_t> &split_set, domain_t &domain) {
		node_idx.clear();

		std::unordered_set<size_t> secondary_splits;

		for (size_t index: split_set) {

			if (!domain.tree[index].is_leaf) continue;

			domain.split(index);

			assert(domain.tree[index].is_leaf == false);
			assert(domain.tree[domain.tree[index].parent_index].tree_level == domain.tree[index].tree_level - 1);

			domain.tree[index].child_indices.for_each([&](size_t child_index) {
				assert(domain.tree[child_index].parent_index == index);
				assert(domain.tree[child_index].is_leaf);
				assert(domain.tree[child_index].tree_level == domain.tree[index].tree_level + 1);

				domain.tree[child_index].can_unsplit = domain.tree[index].can_unsplit;


				node_idx.push_back(child_index);
			});

			domain.tree[index].should_split = false;
		}
	}


	template<typename M>
	void Scene<M>::fix_boundary(domain_t &domain) const {

		std::unordered_set<size_t> new_leaves;

		for (size_t index = 0; index < domain.tree.size(); index++) {
			if (!domain.tree[index].is_active && domain.tree[index].is_leaf) {
				for (int dim = 0; dim < M::NumDims; dim++) {
					for (int dir = 0; dir < 2; dir++) {
						vecn_t neighbor_center = domain.tree[index].cell.center + vecn_t::unit(dim) * (2 * dir - 1) * domain.tree[index].cell.size;

						while (true) {
							size_t neighbor_index = domain.find(neighbor_center, domain.tree[index].tree_level);

							if (neighbor_index && domain.tree[neighbor_index].is_active && domain.tree[neighbor_index].tree_level < domain.tree[index].tree_level) {
								domain.split(neighbor_index);

								if (new_leaves.contains(neighbor_index)) {
									new_leaves.erase(neighbor_index);
								}

								domain.tree[neighbor_index].child_indices.enumerate([&](int ci, size_t child_index) {
									domain.tree[child_index].can_unsplit = domain.tree[neighbor_index].can_unsplit;

									for (int dim = 0; dim < M::NumDims; dim++) {
										for (int dir = 0; dir < 2; dir++) {
											if ((ci & (1 << dim)) == dir << dim)
												domain.tree[child_index].boundaries.get(dim, dir) = domain.tree[neighbor_index].boundaries.get(dim, dir);
										}
									}

									domain.tree[child_index].is_active = true;
									new_leaves.insert(child_index);
								});
							} else {
								break;
							}
						}
					}
				}
			}
		}

		for (size_t index: new_leaves) {
			node_t &n = domain.tree[index];
			n.boundaries.clear();

			for (int dim = 0; dim < M::NumDims; dim++) {
				for (int dir = 0; dir < 2; dir++) {

					vecn_t neighbor_center = n.cell.center + vecn_t::unit(dim) * (2 * dir - 1) * domain.tree[index].cell.size;

					size_t neighbor_index = domain.find(neighbor_center, n.tree_level);

					bool signed_distances_boundary = domain.tree[neighbor_index].is_active != n.is_active;

					scalar_t t;
					bool ray_boundary = raycast(n.cell.center, neighbor_center, t);

					if (signed_distances_boundary || ray_boundary) {
						if (ray_boundary)
							n.boundaries.get(dim, dir) = nearest_boundary(n.cell.get_center());
						else
							n.boundaries.get(dim, dir) = nearest_boundary(n.cell.get_center());
					}
				}
			}
		}

		for (size_t index = 0; index < domain.tree.size(); index++) {
			node_t &n = domain.tree[index];

			if (n.is_active && n.is_leaf) {


				for (int dim = 0; dim < M::NumDims; dim++) {
					for (int dir = 0; dir < 2; dir++) {
						if (n.boundaries.get(dim, dir)) {

							vecn_t neighbor_center = n.cell.center + vecn_t::unit(dim) * (2 * dir - 1) * domain.tree[index].cell.size;

							size_t neighbor_index = domain.find(neighbor_center, n.tree_level);
							if (neighbor_index && !domain.tree[neighbor_index].boundaries.get(dim, 1 - dir)) {
								n.boundaries.get(dim, dir) = std::nullopt;
							}
						}
					}
				}
			}
		}
	}


	template<typename M>
	void Scene<M>::init_interface(scalar_t sd, scalar_t mi, cell_t &cell) const {
		scalar_t alpha = 0.0;

		if (config->objects[mi].type == "droplet") {
			if (sd < 0) alpha = 1.0;
		} else if (config->objects[mi].type == "bubble") {
			if (sd > 0) alpha = 1.0;
			sd *= -1;
		} else {
			return;
		}

		// diffuse alpha field a little
		if (config->ic.diffuse_ic) {
			if (std::abs(sd) < cell.size * config->refinement_radius) {
				alpha = 0.5 * (1.0 - std::tanh(sd / (config->side_length / (1 << config->max_refinement_level))));
			}
		}

		cell.p[0]              = config->rho_ref.value()[0] * alpha;
		cell.p[1]              = config->rho_ref.value()[1] * (1.0 - alpha);
		cell.p[M::NumEqns - 1] = alpha;
	}


	template<typename M>
	void Scene<M>::init_domain(domain_t &domain) {

		node_idx.push_back(1);

		spdlog::info("Constructing initial domain...");

		signed_distances = {-1.0, 0.0};
		nearest_mesh_idx = {0, 0};

		int level = 0;
		while (!node_idx.empty()) {
			spdlog::info("Tree level {}: {} Nodes", level, node_idx.size());
			level++;

			compute_signed_distances(domain);

			std::unordered_set<size_t> split_set;
			compute_node_boundaries(domain, split_set);
			detect_splits(domain, split_set);
			split_detected_nodes(split_set, domain);
		}

		if (config->objects.size() == 0) {
#pragma omp parallel for
			for (int i = 0; i < domain.tree.size(); i++) {
				node_t &node = domain.tree[i];

				if (node.is_leaf) {
					node.is_active = true;
				}
			}
		}

		for (int i = 0; i < 10; i++)
			fix_boundary(domain);

		size_t n_active_cells = 0;
#pragma omp parallel for
		for (size_t index = 0; index < domain.tree.size(); index++) {
			node_t &n = domain.tree[index];

			// check if cell is outside of active domain
			for (size_t dim = 0; dim < M::NumDims; dim++) {
				if (n.cell.center[dim] > 0.5 * config->active_size[dim] || n.cell.center[dim] < -0.5 * config->active_size[dim]) {
					n.is_active = false;
				}
			}

			if (n.is_active && n.is_leaf) {
				n.cell.p = config->ic.x2p(n.cell.get_center());

				if constexpr (M::Multiphase && M::NumDims > 1) {
					init_interface(signed_distances[index], nearest_mesh_idx[index], n.cell);
				}
#pragma omp atomic
				n_active_cells++;
			}
		}


		spdlog::info("Final Tree Size: {} Nodes", domain.tree.size());
		spdlog::info("Active Cells: {}", n_active_cells);
	}


	template<typename M>
	void Scene<M>::visualize_mesh(const domain_t &domain) const {
		std::vector<Eigen::RowVector3d> origins;
		std::vector<Eigen::RowVector3d> normals;
		std::vector<scalar_t> areas;
		std::vector<int> dims;

		for (int i = 0; i < domain.tree.size(); i++) {
			const auto &node = domain.tree[i];

			if (!node.is_leaf || !node.is_active) continue;

			node.boundaries.enumerate(
			    [&](int dim,
			        int dir,
			        const std::optional<integrator::Boundary<M>> &boundary) {
				    if (boundary && node.cell.center[dim] * (2 * dir - 1) < config->active_size[dim] / 2.0) {
					    if (boundary->type == integrator::Boundary<M>::Type::freeslip ||
					        boundary->type == integrator::Boundary<M>::Type::noslip) {
						    origins.push_back(node.cell.face_position(dim, dir).template project<3>().template to_iterable<Eigen::RowVector3d>());
						    normals.push_back(boundary->normal.template project<3>().template to_iterable<Eigen::RowVector3d>());
						    areas.push_back(node.cell.size);
						    dims.push_back(dim);
					    }
				    }
			    });
		}

		Eigen::MatrixXd W(2 * origins.size(), 3);
		Eigen::MatrixXi E(origins.size(), 2);
		Eigen::MatrixXd Q(4 * origins.size(), 3);
		Eigen::MatrixXi F(2 * origins.size(), 3);

		for (int i = 0; i < origins.size(); i++) {
			W.row(2 * i)     = origins[i];
			W.row(2 * i + 1) = origins[i] + normals[i] * 0.005;

			E.row(i) << 2 * i, 2 * i + 1;

			Eigen::RowVector3d unit_1 = Eigen::RowVector3d::Zero();
			Eigen::RowVector3d unit_2 = Eigen::RowVector3d::Zero();

			unit_1[(dims[i] + 1) % 3] = 1.0;
			unit_2[(dims[i] + 2) % 3] = 1.0;

			scalar_t size = areas[i];

			Q.row(4 * i)     = origins[i] + unit_1 * size / 2.0 + unit_2 * size / 2.0;
			Q.row(4 * i + 1) = origins[i] + unit_1 * size / 2.0 - unit_2 * size / 2.0;
			Q.row(4 * i + 2) = origins[i] - unit_1 * size / 2.0 - unit_2 * size / 2.0;
			Q.row(4 * i + 3) = origins[i] - unit_1 * size / 2.0 + unit_2 * size / 2.0;

			F.row(2 * i)     = Eigen::RowVector3i(4 * i, 4 * i + 2, 4 * i + 1);
			F.row(2 * i + 1) = Eigen::RowVector3i(4 * i, 4 * i + 3, 4 * i + 2);
		}

		igl::opengl::glfw::Viewer viewer;
		viewer.data().clear();
		viewer.data().set_mesh(Q, F);
		viewer.data().set_edges(W, E, Eigen::RowVector3d(1.0, 0.0, 0.0));
		viewer.launch();
	}

}    // namespace dacti::scene

template class dacti::scene::Scene<dacti::model::euler<1>>;
template class dacti::scene::Scene<dacti::model::euler<2>>;
template class dacti::scene::Scene<dacti::model::euler<3>>;
template class dacti::scene::Scene<dacti::model::navier_stokes<1>>;
template class dacti::scene::Scene<dacti::model::navier_stokes<2>>;
template class dacti::scene::Scene<dacti::model::navier_stokes<3>>;
template class dacti::scene::Scene<dacti::model::multi<1>>;
template class dacti::scene::Scene<dacti::model::multi<2>>;
template class dacti::scene::Scene<dacti::model::multi<3>>;