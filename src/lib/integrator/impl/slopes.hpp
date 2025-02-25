#pragma once
#include "integrator/integrator.hpp"

namespace dacti::integrator {
	///
	///@brief Compute the slope of a node.
	///		 Compute gradients.
	///
	///@param node_index Index of the node to compute the slope for.
	///
	///This function computes the slope of a node by computing the slopes to both neighbors and applying the slope limiter, in each dimension separately.
	///The slope is stored in the node's slope member.
	///If a boundary is present instead of a neighbor, the boundary condition is used instead of the neighbor's cell.
	///
	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::compute_slope(size_t node_index) {
		auto &node = domain.tree[node_index];
		conv_t pc  = node.cell.p;

		for (size_t dim = 0; dim < NumDims; ++dim) {
			size_t in = node.neighbors.get(dim, 0);
			size_t ip = node.neighbors.get(dim, 1);

			conv_t pn, pp;
			scalar_t xc = 0.0, xn, xp, tn, tp;

			if (in == 0) {    // boundary
				assert(node.boundaries.get(dim, 0).has_value());
				pn = (*node.boundaries.get(dim, 0)).boundary_value(node.cell);
				xn = -node.cell.size;
				tn = node.cell.t;
			} else {
				if (!domain.tree[in].is_leaf) {    // upscaling
					pn = domain.interpolate_p(in);
					tn = domain.interpolate_t(in);
				} else {    // downscaling
					pn = domain.tree[in].cell.p;
					tn = domain.tree[in].cell.t;
				}

				xn = -0.5 * (node.cell.size + domain.tree[in].cell.size);
			}

			if (ip == 0) {    // boundary
				assert(node.boundaries.get(dim, 1).has_value());
				pp = (*node.boundaries.get(dim, 1)).boundary_value(node.cell);
				xp = node.cell.size;
				tp = node.cell.t;
			} else {
				if (!domain.tree[ip].is_leaf) {    // upscaling
					pp = domain.interpolate_p(ip);
					tp = domain.interpolate_t(ip);
				} else {    // downscaling
					pp = domain.tree[ip].cell.p;
					tp = domain.tree[ip].cell.t;
				}

				xp = 0.5 * (node.cell.size + domain.tree[ip].cell.size);
			}

			auto sn = (pc - pn) / (xc - xn);
			auto sp = (pp - pc) / (xp - xc);

			for (int i = 0; i < Model::NumEqns; i++) {
				node.cell.slope[dim][i] = limiter(sn[i], sp[i]);
			}

			if constexpr (Model::NumGrad > 0) {
				node.cell.gradient[dim] = (Model::get_grad_input(pp) - Model::get_grad_input(pn) - (tp - tn) * node.cell.t_slope) / (xp - xn);
			}
		}

		if constexpr (Model::HLLC_solver) {
			Model::compute_rhs(node.cell);
		}
	}


	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::slope_update(int acti_level) {
		ZoneScoped;
#pragma omp parallel for LOOP_SCHEDULE
		for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {
			size_t node_index = acti_clusters[acti_level][i_ind];
			compute_slope(node_index);
		}
	}
}    // namespace dacti::integrator