#pragma once

#include "core/types.hpp"
#include "integrator/domain.hpp"
#include "scene/scene.hpp"
#include <chrono>
#include <omp.h>
#include <span>

#define LOOP_SCHEDULE schedule(guided, 4)

namespace dacti::integrator {

	template<typename Model, typename Limiter, typename ErrorEstimator>
	class Integrator {
	private:
		// for convenience
		using conv_t = Model::conv_t;
		using prim_t = Model::prim_t;
		using flux_t = Model::flux_t;
		using grad_t = Model::grad_t;
		using vecn_t = linalg::vec<scalar_t, Model::NumDims>;
		using node_t = _internal::Node<Model>;

	public:
		static constexpr size_t NumDims = Model::NumDims;
		static constexpr size_t NumMeta = 4;
		static constexpr int MAX_LEVEL  = 16;

		static constexpr std::array<const char *, NumMeta> METADATA_NAMES = {"acti_level", "error", "max_dt", "cfl"};

		scene::Scene<Model> &scene;
		_internal::Domain<Model> domain;
		std::shared_ptr<config::Config<Model>> config;

		Limiter limiter;
		ErrorEstimator error_estimator;

		scalar_t max_error;
		scalar_t min_error;
		size_t max_tree_depth;
		scalar_t max_dt;
		scalar_t max_cfl;

		// ACTI settings
		std::array<std::vector<size_t>, MAX_LEVEL> acti_clusters;    // holds indices of cells active at a given ACTI level
		std::array<std::vector<size_t>, MAX_LEVEL> acti_buffers;     // holds indices of cells that are ahead of their ACTI level
		std::array<int, MAX_LEVEL> acti_counter;
		std::array<scalar_t, MAX_LEVEL> acti_time;

		int n_substeps     = 0;
		int n_global_steps = 0;
		int max_acti_level = 0;

		unsigned long long total_flux_computations = 0;
		unsigned long long total_cell_updates      = 0;

		scalar_t global_dt              = -INFINITY;
		scalar_t runtime_duration       = 0.0;
		scalar_t global_step_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		/**
		 * @brief Quadtree integrator impl constructor
		 * @brief Iterate through all nodes and find leaves. For each leaf:
				- Assign initial conditions and compute initial time step.
				- Compute neighbors and slopes.
				- Add leaf to leaf list.
				- Find globally maximal time step.
		 * @param config General setup for the refinement
		 * @param scene mesh scene and initial conditions
		 */
		Integrator(Limiter limiter,
		           ErrorEstimator error_estimator,
		           scene::Scene<Model> &scene,
		           std::shared_ptr<config::Config<Model>> config) :
		    limiter(limiter),
		    error_estimator(error_estimator),
		    scene(scene),
		    config(config),
		    domain(MAX_LEVEL, config->side_length, config->active_size, config->IsPeriodic) {

			min_error      = config->error_threshold[0];
			max_error      = config->error_threshold[1];
			max_tree_depth = config->max_refinement_level;
			max_dt         = config->max_dt;
			max_cfl        = config->max_cfl;


			spdlog::trace("Initializing domain from initial condition...");

			std::vector<size_t> leaf_indices;

			int n_leaves = 0;

			scene.init_domain(domain);

			/*
				loop over all nodes and assign initial conditions
			*/
			for (int i = 1; i < domain.tree.size(); ++i) {
				auto &node = domain.tree[i];

				if (node.is_leaf) ++n_leaves;

				if (node.is_leaf && node.is_active) {

					assert(node.tree_level == domain.tree[node.parent_index].tree_level + 1);

					leaf_indices.push_back(i);

					node.cell.u = Model::conserved_from_primitive(node.cell.p);
					if constexpr (Model::NumParm > 0)
						Model::compute_parameters(node.cell);

					node.cell.t_slope = linalg::zero<grad_t>();

					node.max_dt = Model::max_delta_time(node.cell);
					global_dt   = std::max(global_dt, node.max_dt);
					node.op     = _internal::creation_op::INIT;
					node.flux   = linalg::zero<flux_t>();
				}
			}

			/*
				Compute initial neighbors and slopes for all leaves
			*/
			spdlog::trace("Computing initial neighbors and slopes...");

			for (int i = 0; i < leaf_indices.size(); ++i) {
				domain.update_neighbors(leaf_indices[i]);
				compute_slope(leaf_indices[i]);
			}

			if constexpr (Model::Multiphase) {
				smooth_alpha(leaf_indices);
				for (int i = 0; i < leaf_indices.size(); ++i) {
					compute_normal(leaf_indices[i]);
				}
				for (int i = 0; i < leaf_indices.size(); ++i) {
					compute_curvature(leaf_indices[i]);
				}
				smooth_curvature(leaf_indices);
			}

			/*
				Initialize for ACTI
			*/
			spdlog::trace("Computing initial time step and ACTI levels...");

			global_dt = std::min(0.9 * global_dt, max_dt);    // global time step is 90% of the maximum time step

			assign_to_acti_clusters(leaf_indices);

			acti_counter.fill(0);
			acti_time.fill(0.0);
			max_acti_level = 0;

			for (int i = 0; i < MAX_LEVEL; i++) {
				if (acti_clusters[i].size() > 0) max_acti_level = std::max(max_acti_level, i);
			}

			runtime_duration = 0.0;
		}

		std::vector<_internal::Node<Model>> get_nodes() const { return domain.tree; }
		_internal::Node<Model> &get_node(size_t index) const { return domain.tree[index]; }
		_internal::Domain<Model> get_domain() const { return domain; }
		size_t get_cell_index(vecn_t pos) const { return domain.find(pos, domain.max_depth); }

		void save_debug_info(int step) { domain.debug(step); }

		void for_each_active_cell(auto action) const {
			int i = 0;
			for (int level = 0; level < MAX_LEVEL; level++) {
				for (size_t node_index: acti_clusters[level]) {
					auto &node = domain.tree[node_index];

					action(i, node);
					i++;
				}
			}

			for (int level = 0; level < MAX_LEVEL; level++) {
				for (size_t node_index: acti_buffers[level]) {
					auto &node = domain.tree[node_index];

					action(i, node);
					i++;
				}
			}
		}

		size_t n_active_cells() const {
			size_t n = 0;
			for (int level = 0; level < MAX_LEVEL; level++) {
				n += acti_clusters[level].size();
			}
			for (int level = 0; level < MAX_LEVEL; level++) {
				n += acti_buffers[level].size();
			}
			return n;
		}

		scalar_t integrate(auto substep_hook, bool signal, bool &force_break);

	private:
		void compute_slope(size_t node_index);
		void compute_normal(size_t node_index);
		void compute_curvature(size_t node_index);
		void smooth_alpha(std::vector<size_t> &cell_cluster);
		void smooth_curvature(std::vector<size_t> &cell_cluster);

		void slope_update(int acti_level);
		void normal_update(int acti_level);
		void curvature_update(int acti_level);
		void flux_update(int acti_level);
		void cell_update(int acti_level, bool signal, bool &force_break);

		void adaptive_refinement(int acti_level);

		void assign_to_acti_clusters(std::span<size_t> node_indices);
		void clear_acti_clusters();
		void flush_acti_buffer(int acti_level);
		void gc();
	};
}    // namespace dacti::integrator

#include "impl/acti_utils.hpp"
#include "impl/adaptive_refinement.hpp"
#include "impl/cell_update.hpp"
#include "impl/flux.hpp"
#include "impl/integrate.hpp"
#include "impl/normal_curvature.hpp"
#include "impl/slopes.hpp"
#include "impl/smooth.hpp"
