#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	/// @brief Flux Computation
	///	    Assumes cell values, slopes and neighbors to have been computed either during initialization or the previous step.
	///     Fluxes are computed using Model::flux for each face associated with the ACTI level and accumulated to the faces' neighboring cells.
	///
	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::flux_update(int acti_level) {

		ZoneScoped;

		scalar_t dt = global_dt / (1 << acti_level);

#pragma omp parallel for LOOP_SCHEDULE
		for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {

			size_t node_index = acti_clusters[acti_level][i_ind];

			auto &node = domain.tree[node_index];

			/*
					Loop through all cell faces. 
				*/
			for (int dim = 0; dim < NumDims; dim++) {
				for (int dir = 0; dir < 2; dir++) {

					size_t neighbor_index = node.neighbors.get(dim, dir);

					typename Model::face_t face{
					    node.cell.center + node.cell.size * vecn_t::unit(dim) * (dir - 0.5),
					    node.cell.size,
					    node.cell.t,         // t0
					    node.cell.t + dt,    // t1
					    dim,
					    dir,
					};
					// #pragma omp atomic
					// 						total_flux_computations++;

					/*
							Fluxes are computed when
							- The face is a boundary
							- neighbor is a leaf &&
								has a lower or equal ACTI level &&
								has a lower tree level || equal tree level but lower ACTI level
						*/
					if (neighbor_index == 0) {
						assert(node.boundaries.get(dim, dir).has_value());

						flux_t flux = (*node.boundaries.get(dim, dir)).boundary_flux(node.cell, face);
						node.flux.fetch_add(-flux);
					} else {
						auto &neighbor = domain.tree[neighbor_index];

						if (neighbor.is_leaf && neighbor.acti_level <= node.acti_level &&
						    (neighbor.tree_level < node.tree_level || neighbor.tree_level == node.tree_level && (dir == 1 || neighbor.acti_level < node.acti_level))) {

							assert(node.is_leaf);
							assert(neighbor.is_leaf);
							assert(neighbor.is_active);

							auto neighbor_copy = neighbor;

							// for periodic boundary conditions, find the correct neighbor
							if (config->IsPeriodic[dim]) {
								if (dir == 1) {
									if (node.cell.center[dim] + node.cell.size > 0.5 * config->active_size[dim]) {
										neighbor_copy.cell.center[dim] += config->active_size[dim];
									}
								} else {
									if (node.cell.center[dim] - node.cell.size < -0.5 * config->active_size[dim]) {
										neighbor_copy.cell.center[dim] -= config->active_size[dim];
									}
								}
							}

							// Fluxes in negative directions are handled by swapping the cells and flipping the direction.
							flux_t flux;
							if (dir == 1) {
								flux = Model::flux(node.cell, neighbor_copy.cell, face);
								node.flux.fetch_add(-flux);
								neighbor.flux.fetch_add(flux);
							} else {
								face.dir = 1;
								flux     = Model::flux(neighbor_copy.cell, node.cell, face);
								node.flux.fetch_add(flux);
								neighbor.flux.fetch_add(-flux);
							}
						}
					}
				}    // loop dirs
			}    // loop dims
		}    // loop cells
	}
}    // namespace dacti::integrator