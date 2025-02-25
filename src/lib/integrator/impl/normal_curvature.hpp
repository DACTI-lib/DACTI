#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::compute_normal(size_t node_index) {
		auto &node = domain.tree[node_index];

		vecn_t normal;

		for (size_t dim = 0; dim < NumDims; ++dim) {
			size_t in = node.neighbors.get(dim, 0);
			size_t ip = node.neighbors.get(dim, 1);

			scalar_t xn, xp;
			scalar_t an, ap;    // surface normals

			if (in == 0) {    // boundary
				an = (*node.boundaries.get(dim, 0)).boundary_value(node.cell)[Model::ia];
				xn = -0.5 * node.cell.size;
			} else {
				if (!domain.tree[in].is_leaf) {    // upscaling
					an = domain.interpolate_k(in, Model::alphas);
				} else {    // downscaling
					an = domain.tree[in].cell.k[Model::alphas];
				}
				xn = -0.5 * (node.cell.size + domain.tree[in].cell.size);
			}

			if (ip == 0) {    // boundary
				ap = (*node.boundaries.get(dim, 1)).boundary_value(node.cell)[Model::ia];
				xp = 0.5 * node.cell.size;
			} else {
				if (!domain.tree[ip].is_leaf) {    // upscaling
					ap = domain.interpolate_k(ip, Model::alphas);
				} else {    // downscaling
					ap = domain.tree[ip].cell.k[Model::alphas];
				}
				xp = 0.5 * (node.cell.size + domain.tree[ip].cell.size);
			}

			normal[dim] = (ap - an) / (xp - xn);
		}

		if (normal.norm() < 1e-10) {
			normal = vecn_t::Zero();
		} else {
			normal = normal / normal.norm();
		}

		for (size_t dim = 0; dim < NumDims; ++dim) {
			node.cell.k[Model::ikn + dim] = normal[dim];
		}
	}


	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::compute_curvature(size_t node_index) {
		auto &node = domain.tree[node_index];

		scalar_t kappa = 0.0;

		// TODO: time corrected stencil might be needed
		for (int dim = 0; dim < NumDims; ++dim) {
			size_t in = node.neighbors.get(dim, 0);
			size_t ip = node.neighbors.get(dim, 1);

			scalar_t xn, xp;
			scalar_t nn, np;    // surface normals

			if (in == 0) {    // boundary
				nn = 0.0;
				xn = -0.5 * node.cell.size;
			} else {
				if (!domain.tree[in].is_leaf) {    // upscaling
					nn = domain.interpolate_k(in, Model::ikn + dim);
				} else {    // downscaling
					nn = domain.tree[in].cell.k[Model::ikn + dim];
				}

				xn = -0.5 * (node.cell.size + domain.tree[in].cell.size);
			}

			if (ip == 0) {    // boundary
				np = 0.0;
				xp = 0.5 * node.cell.size;
			} else {
				if (!domain.tree[ip].is_leaf) {    // upscaling
					np = domain.interpolate_k(ip, Model::ikn + dim);
				} else {    // downscaling
					np = domain.tree[ip].cell.k[Model::ikn + dim];
				}

				xp = 0.5 * (node.cell.size + domain.tree[ip].cell.size);
			}

			kappa -= (np - nn) / (xp - xn);
		}

		node.cell.k[Model::kappa] = kappa;
	}

	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::normal_update(int acti_level) {
		ZoneScoped;
#pragma omp parallel for LOOP_SCHEDULE
		for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {
			size_t node_index = acti_clusters[acti_level][i_ind];
			compute_normal(node_index);
		}
	}

	template<typename Model, typename Limiter, typename ErrorEstimator>
	void Integrator<Model, Limiter, ErrorEstimator>::curvature_update(int acti_level) {
		ZoneScoped;
#pragma omp parallel for LOOP_SCHEDULE
		for (size_t i_ind = 0; i_ind < acti_clusters[acti_level].size(); i_ind++) {
			size_t node_index = acti_clusters[acti_level][i_ind];
			compute_curvature(node_index);
		}
	}
}    // namespace dacti::integrator