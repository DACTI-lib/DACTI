#pragma once
#include "core/types.hpp"
#include "integrator/cell.hpp"
#include "integrator/node.hpp"
#include <dacti.hpp>
#include <fmt/core.h>
#include <fstream>


namespace dacti::integrator::_internal {

	constexpr size_t NULL_NODE = 0;

	template<typename Model>
	struct Domain {

		static constexpr size_t NumDims     = Model::NumDims;
		static constexpr size_t NumChildren = 1 << NumDims;

		using vecn_t = linalg::vec<scalar_t, NumDims>;
		using node_t = Node<Model>;

		const size_t root_index = 1;
		size_t max_depth;

		scalar_t side_length;
		vecn_t active_size;
		std::array<bool, NumDims> IsPeriodic;

		std::vector<node_t> tree;

		Domain(size_t max_depth,
		       scalar_t side_length,
		       vecn_t active_size,
		       std::array<bool, NumDims> IsPeriodic) :
		    max_depth(max_depth),
		    side_length(side_length),
		    active_size(active_size),
		    IsPeriodic(IsPeriodic) {

			node_t root;
			root.parent_index  = 0;
			root.is_leaf       = true;
			root.acti_level    = -1;
			root.tree_level    = 0;
			root.cell.center   = vecn_t::Zero();
			root.cell.size     = side_length;
			root.cell.t        = 0.0;
			root.is_active     = true;
			root.is_garbage    = false;
			root.can_split     = true;
			root.can_unsplit   = false;
			root.should_split  = false;
			root.creation_time = 0;
			root.op            = creation_op::INIT;

			root.neighbors.fill(0);
			root.boundaries.fill(std::nullopt);
			root.child_indices.fill(0);

			tree.push_back(node_t{.tree_level = -1});    // invalid node -> corresponds to index 0
			tree.push_back(root);
		}

		size_t find(vecn_t p, int max_level) const {

			for (int i = 0; i < NumDims; i++) {
				if (p[i] < -0.5 * active_size[i] || p[i] > 0.5 * active_size[i]) {
					if (IsPeriodic[i]) {
						if (p[i] < -0.5 * active_size[i])
							p[i] += active_size[i];
						else
							p[i] -= active_size[i];
					} else {
						return 0;
					}
				}
			}
			vecn_t x = 2.0 * p / tree[root_index].cell.size;

			return _find(x, root_index, max_level);
		}

		Model::prim_t interpolate_p(size_t index) {
			assert(index != 0);
			assert(!tree[index].is_garbage && "Cannot interpolate garbage node.");

			const node_t &node = tree[index];

			if (node.is_leaf) {
				return node.cell.get_p();
			} else {
				auto f = Model::prim_t::Zero();
				int n  = 0;

				node.child_indices.for_each([&](size_t child_index) {
					if (tree[child_index].is_active) {
						f += interpolate_p(child_index);
						n++;
					}
				});
				return f / n;
			}
		}

		scalar_t interpolate_k(size_t index, size_t i) const {
			assert(index != 0);
			assert(!tree[index].is_garbage && "Cannot interpolate garbage node.");

			const node_t &node = tree[index];

			if (node.is_leaf) {
				return node.cell.get_k()[i];
			} else {
				scalar_t f = 0.0;
				int n      = 0;

				node.child_indices.for_each([&](size_t child_index) {
					if (tree[child_index].is_active) {
						f += interpolate_k(child_index, i);
						n++;
					}
				});
				return f / n;
			}
		}

		scalar_t interpolate_t(size_t index) {
			assert(index != 0);
			assert(!tree[index].is_garbage && "Cannot interpolate garbage node.");

			const node_t &node = tree[index];

			if (node.is_leaf) {
				return node.cell.get_t();
			} else {
				scalar_t t = 0.0;
				int n      = 0;

				node.child_indices.for_each([&](size_t child_index) {
					if (tree[child_index].is_active) {
						t += interpolate_t(child_index);
						n++;
					}
				});
				return t / n;
			}
		}

		Cell<Model> interpolated_cell(size_t index) const {
			if (tree[index].is_leaf) {
				return tree[index].cell;
			} else {

				Cell<Model> cell = tree[index].cell;

				cell.t       = 0;
				cell.t_slope = Model::grad_t::Zero();
				cell.u       = Model::conv_t::Zero();
				cell.k       = Model::parm_t::Zero();

				tree[index].child_indices.for_each([&](size_t child_index) {
					auto child_cell  = interpolated_cell(child_index);
					cell.t          += child_cell.t;
					cell.u          += child_cell.u;
					cell.k          += child_cell.k;
					cell.gradient   += child_cell.gradient;
					cell.t_slope    += child_cell.t_slope;
				});

				cell.t        /= NumChildren;
				cell.t_slope  /= NumChildren;
				cell.u        /= NumChildren;
				cell.p         = Model::primitive_from_conserved(cell.u);
				cell.k        /= NumChildren;
				cell.gradient /= NumChildren;

				return cell;
			}
		}

		void visit_leaves(size_t index, const std::function<void(size_t)> &visitor) {
			if (tree[index].is_leaf) {
				visitor(index);
			} else {
				tree[index].child_indices.for_each([&](size_t child_index) {
					visit_leaves(child_index, visitor);
				});
			}
		}

		void split(size_t parent_index) {

			assert(tree[parent_index].is_leaf && "Cannot split non-leaf node.");

			auto parent = tree[parent_index];


			parent.child_indices.enumerate([&](size_t i, size_t child_index) {
				node_t child;

				child.parent_index  = parent_index;
				child.is_leaf       = true;
				child.acti_level    = parent.acti_level;
				child.tree_level    = parent.tree_level + 1;
				child.cell.center   = parent.cell.center + 0.25 * parent.cell.size * vecn_t::quadrant(i);
				child.cell.size     = 0.5 * parent.cell.size;
				child.cell.t        = parent.cell.t;
				child.cell.p        = parent.cell.get_p(child.cell.center, child.cell.t);
				child.cell.u        = Model::conserved_from_primitive(child.cell.p);
				child.cell.k        = parent.cell.get_k(child.cell.center, child.cell.t);
				child.cell.slope    = parent.cell.slope;
				child.cell.gradient = parent.cell.gradient;
				child.cell.t_slope  = parent.cell.t_slope;
				child.error         = -1.0;
				child.should_split  = false;
				child.is_garbage    = false;

				child.can_split   = true;
				child.can_unsplit = true;

				child.creation_time = parent.cell.t;
				child.op            = creation_op::SPLIT;

				child.neighbors     = Neighbor<size_t, NumDims>{};
				child.boundaries    = Neighbor<std::optional<Boundary<Model>>, NumDims>{};
				child.child_indices = Children<size_t, NumDims>{};

				child.neighbors.fill(0);
				child.child_indices.fill(0);

				if (child_index != 0) {

					assert(child_index < tree.size());
					auto reused_child   = tree[child_index];
					child.child_indices = reused_child.child_indices;
					tree[child_index]   = child;
				} else {
					tree.push_back(child);
					parent.child_indices[i] = tree.size() - 1;
				}
			});

			parent.is_leaf = false;

			tree[parent_index] = parent;
		}

		void update_neighbors(size_t node_index) {

			assert(node_index != 0 && "Cannot update neighbors of invalid node.");

			auto &node = tree[node_index];

			if (!node.is_leaf) {
				node.child_indices.for_each([&](size_t child_index) {
					update_neighbors(child_index);
				});
				return;
			}

			if (!node.is_active)
				return;

			for (int dim = 0; dim < NumDims; dim++) {
				for (int dir = 0; dir < 2; dir++) {
					if (node.boundaries.get(dim, dir).has_value())
						continue;

					vecn_t location = node.cell.center + vecn_t::unit(dim) * node.cell.size * (2 * dir - 1.0);

					size_t neighbor_index = find(location, node.tree_level);


					if ((!neighbor_index || !tree[neighbor_index].is_active)) {
#ifndef NDEBUG
						spdlog::error("Inactive neighbor at index {}.", neighbor_index);
						spdlog::error("Neighbor position: {} {} {}", location[0], location[1], location[2]);
						spdlog::error("Actual Neighbor position: {} {} {}", tree[neighbor_index].cell.center[0], tree[neighbor_index].cell.center[1], tree[neighbor_index].cell.center[2]);
						spdlog::error("Node position: {} {} {}", node.cell.center[0], node.cell.center[1], node.cell.center[2]);
#endif
						continue;
					}


					node.neighbors.get(dim, dir) = neighbor_index;

					if (tree[neighbor_index].tree_level < node.tree_level - 1) {
						tree[neighbor_index].should_split = true;
					}
				}
			}
		}


		void unsplit(size_t parent_index) {
			assert(!tree[parent_index].is_leaf && "Cannot unsplit leaf node.");
			assert(parent_index != 0 && "Cannot unsplit invalid node." && parent_index < tree.size());

			auto &parent = tree[parent_index];
			auto cell    = interpolated_cell(parent_index);

			scalar_t error = 0.0;

			parent.child_indices.for_each([&](size_t child_index) {
				//assert(tree[child_index].is_leaf && "Cannot unsplit complex node.");
				error                        += tree[child_index].error;
				tree[child_index].is_garbage  = true;
				tree[child_index].is_leaf     = true;
			});

			parent.neighbors.fill(0);

			parent.cell          = cell;
			parent.is_leaf       = true;
			parent.error         = error;
			parent.flux          = Model::flux_t::Zero();
			parent.creation_time = cell.t;
			//parent.op = creation_op::UNSPLIT;

			parent.is_active = true;

			parent.acti_level = -1;
			parent.can_split  = true;
		}

		void gc() {
			//std::vector<node> new_tree;
			std::vector<size_t> old_to_new(tree.size(), 0);

			int new_tree_end = 1;

			for (int i = 1; i < tree.size(); ++i) {
				if (!tree[i].is_garbage) {
					tree[new_tree_end] = tree[i];
					old_to_new[i]      = new_tree_end;
					new_tree_end++;
				} else {
					old_to_new[i] = 0;
				}
			}

			tree.resize(new_tree_end);

			for (int i = 1; i < tree.size(); ++i) {
				tree[i].parent_index = old_to_new[tree[i].parent_index];

				tree[i].child_indices = tree[i].child_indices.template map<size_t>([&](size_t child_index) {
					return old_to_new[child_index];
				});

				tree[i].neighbors = tree[i].neighbors.template map<size_t>([&](size_t neighbor_index) {
					return old_to_new[neighbor_index];
				});
			}
		}

		void debug(int step) const {
			//std::ofstream nfile("nodes.csv");
			//std::ofstream ffile("faces.csv");
			//std::ofstream bfile("boundaries.csv");

			std::ofstream nfile(fmt::format("nodes_{:04}.csv", step));
			std::ofstream ffile(fmt::format("interior_faces_{:04}.csv", step));
			std::ofstream bfile(fmt::format("boundary_faces_{:04}.csv", step));

			ffile << "x0,y0,x1,y1\n";
			nfile << "x,y,acti_level,tree_level,leaf\n";
			bfile << "x0,y0,x1,y1,nx,ny\n";

			for (int i = 1; i < tree.size(); i++) {
				const auto &n = tree[i];

				if (n.is_active && !n.is_garbage) {
					nfile << n.cell.center[0] << "," << n.cell.center[1] << "," << n.acti_level << "," << n.tree_level << "," << n.is_leaf << "\n";


					if (n.is_leaf) {
						for (int dim = 0; dim < NumDims; ++dim) {
							for (int dir = 0; dir < 2; ++dir) {
								size_t neighbor_index = n.neighbors.get(dim, dir);

								if (neighbor_index != 0 && tree[neighbor_index].is_active) {
									const auto &neighbor = tree[neighbor_index];

									vecn_t x0 = n.cell.center + 0.5 * vecn_t::unit(dim) * n.cell.size * (dir - 0.5);
									vecn_t x1 = n.cell.center + 0.75 * vecn_t::unit(dim) * n.cell.size * (2 * dir - 1.0);

									ffile << x0[0] << "," << x0[1] << "," << x1[0] << "," << x1[1] << "\n";
								}

								if (n.boundaries.get(dim, dir).has_value()) {

									vecn_t p1 = n.cell.center + vecn_t::unit(dim) * n.cell.size * (2 * dir - 1);

									bfile << n.cell.center[0] << "," << n.cell.center[1] << "," << p1[0] << "," << p1[1] << "," << n.boundaries.get(dim, dir)->normal[0] << "," << n.boundaries.get(dim, dir)->normal[1] << "\n";
								}
							}
						}
					}
				}
			}
		}


	private:
		size_t _find(vecn_t p, size_t index, int max_level) const {


			const node_t &node = tree[index];
			if (node.is_leaf || node.tree_level == max_level) {
				return index;
			}

			size_t i = p.cwise_op_bits([](double x) { return x > 0.0; });
			return _find((p - 0.5 * vecn_t::quadrant(i)) * 2.0, node.child_indices[i], max_level);
		}
	};

}    // namespace dacti::integrator::_internal
