#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::adaptive_refinement(int acti_level) {
		ZoneScoped;
		scalar_t dt = global_dt / (1 << acti_level);

		/* Error Estimation 
				The error estimation step loops through all cells in the ACTI level and estimates the error due to the spatial discretization.
				The error is stored in the node's error member.
			*/

		std::vector<size_t> split_cells;
		std::vector<size_t> maybe_unsplit;
		{
			ZoneScopedN("Error Estimation");
#pragma omp parallel
			{
				std::vector<size_t> split_cells_private;
				std::vector<size_t> maybe_unsplit_private;

				split_cells_private.reserve(acti_clusters[acti_level].size() / omp_get_num_threads() / 4);
				maybe_unsplit_private.reserve(acti_clusters[acti_level].size() / omp_get_num_threads() / 4);

#pragma omp for LOOP_SCHEDULE
				for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {

					size_t node_index = acti_clusters[acti_level][i_ind];
					auto &node        = domain.tree[node_index];

					/* Many error estimators require neighbor information. */
					if constexpr (decltype(error_estimator)::RequiresNeighborhood) {
						_internal::Neighbor<std::optional<prim_t>, NumDims> neighborhood;

						for (int dim = 0; dim < NumDims; dim++) {
							for (int dir = 0; dir < 2; dir++) {
								size_t neighbor_index = node.neighbors.get(dim, dir);

								if (neighbor_index == 0) {
									neighborhood.get(dim, dir) = std::nullopt;
								} else {
									if (!domain.tree[neighbor_index].is_leaf) {
										neighborhood.get(dim, dir) = domain.interpolate_p(neighbor_index);
									} else {
										if (domain.tree[neighbor_index].tree_level < node.tree_level) {
											vecn_t x = node.cell.get_center() + node.cell.size * vecn_t::unit(dim) * (2 * dir - 1.0);

											if (config->IsPeriodic[dim] && dir == 0 && x[dim] < -0.5 * config->active_size[dim]) {
												x[dim] += config->active_size[dim];
											}

											if (config->IsPeriodic[dim] && dir == 1 && x[dim] > 0.5 * config->active_size[dim]) {
												x[dim] -= config->active_size[dim];
											}

											neighborhood.get(dim, dir) = domain.tree[neighbor_index].cell.get_p(x, node.cell.t);
										} else {
											neighborhood.get(dim, dir) = domain.tree[neighbor_index].cell.p;
										}
									}
								}
							}
						}

						node.error = error_estimator.estimate(node.cell, neighborhood);
					} else {
						node.error = error_estimator.estimate(node.cell);
					}

					int max_depth = domain.max_depth;

					if (node.can_split)
						node.can_split = scene.can_split(node.cell.center, node.tree_level);

					if ((node.error > max_error && node.can_split) || node.should_split) {
						split_cells_private.push_back(node_index);
					}

					if (node.error < min_error && node.can_unsplit && node.tree_level > 0 && acti_counter[acti_level] == 1) {
						maybe_unsplit_private.push_back(node_index);
					}
				}

#pragma omp critical
				{
					ZoneScopedN("Critical");
					split_cells.insert(split_cells.end(), split_cells_private.begin(), split_cells_private.end());
					maybe_unsplit.insert(maybe_unsplit.end(), maybe_unsplit_private.begin(), maybe_unsplit_private.end());
				}
			}
		}    // Error Estimation


		std::vector<size_t> new_acti_cluster;
		std::unordered_set<size_t> unsplit_parents;
		std::unordered_set<size_t> unsplit_children;
		std::unordered_set<size_t> invalid_neighborhoods;


		/* Splitting / Unsplitting */
		/* 
				Loops through all cells in the ACTI level and checks if they should be split or unsplit.
				Neighboring cells should never differ by more than one spatial refinement level.
				Cells are split if their error is too large and they are not at the maximum tree depth and if the above criterion would not be violated.
				Cells are unsplit if the converse is true.
			*/
		{
			ZoneScopedN("Spatial Refinement");
			{
				ZoneScopedN("Detection");
				//#pragma omp parallel
				{

					std::vector<size_t> unsplit_parents_private;
					std::vector<size_t> unsplit_children_private;

					//#pragma omp for
					for (size_t node_index: maybe_unsplit) {
						auto &node = domain.tree[node_index];

						if (!node.can_unsplit) continue;

						bool unsplit = true;

						node.neighbors.for_each(
						    [&](size_t neighbor_index) {
							    if (neighbor_index != 0) {
								    if (!domain.tree[neighbor_index].is_leaf || domain.tree[neighbor_index].acti_level > acti_level) {
									    unsplit = false;
								    }
							    }
						    });

						if (unsplit) {
							unsplit_parents_private.push_back(node.parent_index);
							unsplit_children_private.push_back(node_index);
						}
					}

					//#pragma omp critical
					{
						unsplit_parents.insert(unsplit_parents_private.begin(), unsplit_parents_private.end());
						unsplit_children.insert(unsplit_children_private.begin(), unsplit_children_private.end());
					}
				}
			}

			{
				ZoneScopedN("Application");

				for (size_t node_index: split_cells) {

					bool ready_to_split = true;
					domain.tree[node_index].neighbors.for_each(
					    [&](size_t neighbor_index) {
						    if (neighbor_index != 0) {
							    if (domain.tree[neighbor_index].tree_level < domain.tree[node_index].tree_level - 1) {
								    ready_to_split = false;
							    }
						    }
					    });
					if (!ready_to_split) continue;

					auto old_boundaries = domain.tree[node_index].boundaries;

					compute_slope(node_index);

					domain.split(node_index);


					auto &node = domain.tree[node_index];

					node.should_split = false;

					node.child_indices.enumerate(
					    [&](int ci, size_t child_index) {
						    auto &child = domain.tree[child_index];

						    child.is_active = true;    // TODO: only works if no boundary cells are

						    for (int dim = 0; dim < NumDims; dim++) {
							    for (int dir = 0; dir < 2; dir++) {
								    if ((ci & (1 << dim)) == dir << dim)
									    child.boundaries.get(dim, dir) = node.boundaries.get(dim, dir);
							    }
						    }


						    child.acti_level = acti_level;
						    acti_clusters[acti_level].push_back(child_index);
						    invalid_neighborhoods.insert(child_index);
					    });
				}

				for (size_t parent_index: unsplit_parents) {
					bool ready_to_unsplit = true;

					auto &parent = domain.tree[parent_index];

					parent.child_indices.for_each(
					    [&](size_t child_index) {
						    if (!unsplit_children.contains(child_index)) {
							    ready_to_unsplit = false;
						    }
					    });


					if (ready_to_unsplit) {
						domain.unsplit(parent_index);

						acti_clusters[acti_level].push_back(parent_index);
						parent.acti_level = acti_level;
						parent.is_active  = true;

						invalid_neighborhoods.insert(parent_index);
					}
				}
			}


			{
				ZoneScopedN("Neighbor Update");

				std::unordered_set<size_t> secondary_invalid_neighborhoods;

				std::vector<size_t> invalid_neighborhoods_vector{invalid_neighborhoods.begin(), invalid_neighborhoods.end()};

#pragma omp parallel
				{
					std::vector<size_t> secondary_invalid_neighborhoods_private;

#pragma omp for
					for (size_t node_index: invalid_neighborhoods_vector) {

						domain.update_neighbors(node_index);
						// compute_slope(node_index);// TODO: not technically necessary

						for (int dim = 0; dim < NumDims; dim++) {
							for (int dir = 0; dir < 2; dir++) {
								size_t neighbor_index = domain.tree[node_index].neighbors.get(dim, dir);
								if (neighbor_index != 0 && (domain.tree[neighbor_index].is_active || !domain.tree[neighbor_index].is_leaf)) {
									domain.update_neighbors(neighbor_index);
								}
							}
						}
					}

#pragma omp critical
					{
						secondary_invalid_neighborhoods.insert(secondary_invalid_neighborhoods_private.begin(), secondary_invalid_neighborhoods_private.end());
					}
				}

				for (size_t node_index: secondary_invalid_neighborhoods) {
					if (!invalid_neighborhoods.contains(node_index))
						domain.update_neighbors(node_index);
				}
			}
		}    // spatial refinement

		// Temporal refinement
		{
			ZoneScopedN("ACTI reassignment");
#pragma omp parallel
			{

				std::vector<size_t> new_acti_cluster_private;
				std::vector<size_t> lowered_cells_private;
				std::vector<size_t> raised_cells_private;

#pragma omp for
				for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {
					size_t node_index = acti_clusters[acti_level][i_ind];
					auto &node        = domain.tree[node_index];

					scalar_t new_max_dt = Model::max_delta_time(node.cell);

					node.max_dt = new_max_dt;
					node.cfl    = dt / node.max_dt;


					if (!node.is_leaf || node.is_garbage) {
						continue;
					}

					/* ACTI counter of 1 means that level is synchronized with level n-1, and has been updated twice*/
					if (node.cfl < 0.5 * max_cfl && acti_counter[acti_level] == 1 && acti_level > 0) {
						lowered_cells_private.push_back(node_index);
						node.cfl *= 2;
					} else if (node.cfl > max_cfl && acti_level < MAX_LEVEL - 1) {

						int acti_raise = std::log2(node.cfl / max_cfl) + 1;

						if (acti_level + acti_raise < MAX_LEVEL) {
							raised_cells_private.push_back(node_index);
							node.acti_level  = acti_level + acti_raise;
							node.cfl        /= 1 << acti_raise;
						}
					} else {
						new_acti_cluster_private.push_back(node_index);
					}
				}


#pragma omp critical
				{
					new_acti_cluster.insert(new_acti_cluster.end(), new_acti_cluster_private.begin(), new_acti_cluster_private.end());
					if (acti_level > 0) acti_buffers[acti_level - 1].insert(acti_buffers[acti_level - 1].end(), lowered_cells_private.begin(), lowered_cells_private.end());


					for (int i = 0; i < raised_cells_private.size(); i++) {
						int acti_level = domain.tree[raised_cells_private[i]].acti_level;
						acti_clusters[acti_level].push_back(raised_cells_private[i]);
					}
				}
			}

			acti_clusters[acti_level] = new_acti_cluster;
		}    // Temporal refinement
	}
}    // namespace dacti::integrator