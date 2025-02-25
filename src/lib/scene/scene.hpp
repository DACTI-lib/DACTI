#pragma once

#include "config/config.hpp"
#include "integrator/boundary.hpp"
#include "integrator/domain.hpp"
#include <unordered_set>

namespace dacti::scene {

	template<typename M>
	class Scene {
	private:
		using vecn_t   = linalg::vec<scalar_t, M::NumDims>;
		using domain_t = integrator::_internal::Domain<M>;
		using node_t   = integrator::_internal::Node<M>;
		using cell_t   = integrator::_internal::Cell<M>;

		std::vector<size_t> node_idx;
		std::vector<size_t> nearest_mesh_idx;
		std::vector<scalar_t> signed_distances;

	public:
		const std::shared_ptr<config::Config<M>> config;

		// constructor
		Scene(std::shared_ptr<config::Config<M>> config) :
		    config(config) {}


		integrator::Boundary<M> nearest_boundary(vecn_t point) const;

		bool raycast(vecn_t a, vecn_t b, scalar_t &t) const;

		void compute_signed_distances(domain_t &domain, bool set_active = true);

		void compute_node_boundaries(domain_t &domain, std::unordered_set<size_t> &split_set, bool raytrace = true);

		void detect_splits(domain_t &domain, std::unordered_set<size_t> &split_set);

		void split_detected_nodes(std::unordered_set<size_t> &split_set, domain_t &domain);

		void fix_boundary(domain_t &domain) const;

		void init_interface(scalar_t sd, scalar_t mi, cell_t &cell) const;

		void init_domain(domain_t &domain);

		void visualize_mesh(const domain_t &domain) const;

		bool can_split(vecn_t x, size_t current_level) {
			if (current_level < config->max_refinement_level)
				return true;

			for (const auto &zone: config->refine_zones) {
				if (current_level < zone.refinement_level && zone.contains(x.template project<3>())) {
					return true;
				}
			}

			return false;
		}
	};

}    // namespace dacti::scene