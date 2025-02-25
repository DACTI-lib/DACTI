#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	/* Cell Update
		* The cell update step loops through all cells in the ACTI level and updates their values using the accumulated fluxes
		*/
	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::cell_update(int acti_level, bool signal, bool &force_break) {
		ZoneScoped;
		scalar_t dt = global_dt / (1 << acti_level);

#pragma omp parallel for LOOP_SCHEDULE
		for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {
			size_t node_index = acti_clusters[acti_level][i_ind];
			auto &node        = domain.tree[node_index];

			grad_t grad_input_old;
			node.cell.t_slope = linalg::zero<grad_t>();
			if constexpr (Model::NumGrad > 0)
				grad_input_old = Model::get_grad_input(node.cell.p);

			/*
					cell updates
				*/
			flux_t du = node.flux / node.cell.volume();
			for (size_t k = 0; k < Model::NumEqns; k++) {
				node.cell.u[k] += du[k];
			}

			// Hold droplet still
			if constexpr (Model::Multiphase) {
				if (!signal) {
					if (node.cell.u[Model::ia] > 0.001) {
						for (size_t k = Model::iu; k < Model::iu + NumDims; k++) {
							node.cell.u[k] = 0.0;
						}
					}
				}
			}

			/*
					add source terms
				*/
			if constexpr (Model::HasSource) {
				node.cell.u += Model::source(node.cell, dt, du);
			}

			node.cell.p = Model::primitive_from_conserved(node.cell.u);

			if (isnan(node.cell.p[Model::ip]) || isnan(node.cell.p[0])) {
#pragma omp critical
				{
					force_break = true;
				}
			}

			/*
					limit unphysical values
				*/
			if constexpr (Model::Multiphase) {
				node.cell.p[0] = std::max(node.cell.p[0], 1e-3);
				node.cell.p[1] = std::max(node.cell.p[1], 1e-6);
			} else {
				node.cell.p[0] = std::max(node.cell.p[0], 1e-3);
			}
			node.cell.p[Model::ip] = std::max(node.cell.p[Model::ip], 1e-3);
			node.cell.u            = Model::conserved_from_primitive(node.cell.p);

			/*
					compute t_slope for gradient computation
				*/
			if constexpr (Model::NumGrad > 0) {
				node.cell.t_slope = (Model::get_grad_input(node.cell.p) - grad_input_old) / dt;
			}

			/*
					update parameters
				*/
			if constexpr (Model::NumParm > 0)
				Model::compute_parameters(node.cell);

			node.flux    = linalg::zero<flux_t>();
			node.cell.t += dt;

			// #pragma omp atomic
			// 				total_cell_updates++;
		}
	}
}    // namespace dacti::integrator