#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	static constexpr size_t a_smooth_iter = 3;
	static constexpr size_t k_smooth_iter = 1;

	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::smooth_alpha(std::vector<size_t> &cell_cluster) {
		ZoneScoped;
		std::vector<scalar_t> alpha_smooth(cell_cluster.size(), 0);

		for (size_t iter = 0; iter < a_smooth_iter; iter++) {
#pragma omp parallel for LOOP_SCHEDULE
			for (size_t i_ind = 0; i_ind < cell_cluster.size(); i_ind++) {
				auto &node = domain.tree[cell_cluster[i_ind]];

				for (size_t dim = 0; dim < NumDims; dim++) {
					size_t in = node.neighbors.get(dim, 0);
					size_t ip = node.neighbors.get(dim, 1);

					scalar_t an, ap;
					if (in == 0) {    // boundary
						an = (*node.boundaries.get(dim, 0)).boundary_value(node.cell)[Model::ia];
					} else {
						if (!domain.tree[in].is_leaf) {    // upscaling
							an = domain.interpolate_k(in, Model::alphas);
						} else {    // downscaling
							an = domain.tree[in].cell.k[Model::alphas];
						}
					}

					if (ip == 0) {    // boundary
						ap = (*node.boundaries.get(dim, 1)).boundary_value(node.cell)[Model::ia];
					} else {
						if (!domain.tree[ip].is_leaf) {    // upscaling
							ap = domain.interpolate_k(ip, Model::alphas);
						} else {    // downscaling
							ap = domain.tree[ip].cell.k[Model::alphas];
						}
					}
					alpha_smooth[i_ind] += 0.5 * (an + ap) + node.cell.k[Model::alphas];
					// alpha_smooth[i_ind] += an + ap;
				}
				// alpha_smooth[i_ind] = 0.5 * alpha_smooth[i_ind] / (2*NumDims) + 0.5 * node.cell.k[Model::alphas];
				alpha_smooth[i_ind] = alpha_smooth[i_ind] / (2 * NumDims);

				// scalar_t D = 1.0 / (2 * NumDims);
				// alpha_smooth[i_ind] = (1.0 - 2 * NumDims * D) * node.cell.k[Model::alphas] + D * alpha_smooth[i_ind];
			}

#pragma omp parallel for LOOP_SCHEDULE
			for (size_t i_ind = 0; i_ind < cell_cluster.size(); i_ind++) {
				auto &node                 = domain.tree[cell_cluster[i_ind]];
				node.cell.k[Model::alphas] = alpha_smooth[i_ind];
				alpha_smooth[i_ind]        = 0.0;
			}
		}
	}


	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::smooth_curvature(std::vector<size_t> &cell_cluster) {
		ZoneScoped;
		std::vector<scalar_t> kappa_smooth(cell_cluster.size(), 0);

		for (size_t iter = 0; iter < k_smooth_iter; iter++) {
#pragma omp parallel for LOOP_SCHEDULE
			for (size_t i_ind = 0; i_ind < cell_cluster.size(); i_ind++) {
				auto &node  = domain.tree[cell_cluster[i_ind]];
				scalar_t kc = node.cell.k[Model::kappa];
				scalar_t wc = std::pow(node.cell.k[Model::alphas] * (1 - node.cell.k[Model::alphas]), 2.0);
				// scalar_t wc = std::sqrt(node.cell.p[Model::ia] * (1 - node.cell.p[Model::ia]));
				scalar_t w = 0.0;

				// kappa_smooth[i_ind] = kc*wc;
				// w = wc;

				for (size_t dim = 0; dim < NumDims; dim++) {
					size_t in = node.neighbors.get(dim, 0);
					size_t ip = node.neighbors.get(dim, 1);

					scalar_t kn, kp;
					scalar_t an, ap;
					if (in == 0) {    // boundary
						kn = 0.0;
						an = (*node.boundaries.get(dim, 0)).boundary_value(node.cell)[Model::ia];
					} else {
						if (!domain.tree[in].is_leaf) {    // upscaling
							kn = domain.interpolate_k(in, Model::kappa);
							// an = domain.interpolate_k(in, Model::alphas);
							an = domain.interpolate_p(in)[Model::ia];
						} else {    // downscaling
							kn = domain.tree[in].cell.k[Model::kappa];
							// an = domain.tree[in].cell.k[Model::alphas];
							an = domain.tree[in].cell.p[Model::ia];
						}
					}

					if (ip == 0) {    // boundary
						kp = 0.0;
						ap = (*node.boundaries.get(dim, 1)).boundary_value(node.cell)[Model::ia];
					} else {
						if (!domain.tree[ip].is_leaf) {    // upscaling
							kp = domain.interpolate_k(ip, Model::kappa);
							// ap = domain.interpolate_k(ip, Model::alphas);
							ap = domain.interpolate_p(ip)[Model::ia];
						} else {    // downscaling
							kp = domain.tree[ip].cell.k[Model::kappa];
							// ap = domain.tree[ip].cell.k[Model::alphas];
							ap = domain.tree[ip].cell.p[Model::ia];
						}
					}

					// scalar_t wn = std::sqrt(an * (1 - an));
					// scalar_t wp = std::sqrt(ap * (1 - ap));

					scalar_t wn = std::pow(an * (1 - an), 2.0);
					scalar_t wp = std::pow(ap * (1 - ap), 2.0);

					kappa_smooth[i_ind] += 0.5 * (wn * kn + wp * kp) + kc * wc;
					w                   += 0.5 * (wn + wp) + wc;

					// kappa_smooth[i_ind] += wn * kn + wp * kp;
					// w += wn + wp;
				}
				if (w > 1e-6)
					// kappa_smooth[i_ind] = (1.0 - 2.0 * wc) * kappa_smooth[i_ind] / w + 2.0 * wc * kc;
					kappa_smooth[i_ind] = kappa_smooth[i_ind] / w;
				else
					kappa_smooth[i_ind] = 0.0;
			}

#pragma omp parallel for LOOP_SCHEDULE
			for (size_t i_ind = 0; i_ind < cell_cluster.size(); i_ind++) {
				auto &node                = domain.tree[cell_cluster[i_ind]];
				node.cell.k[Model::kappa] = kappa_smooth[i_ind];
				kappa_smooth[i_ind]       = 0.0;
			}
		}
	}
}    // namespace dacti::integrator