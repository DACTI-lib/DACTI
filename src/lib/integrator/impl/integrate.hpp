#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	/**
		 * @brief Runs the integrator.
		 * 
		 * Terminates when the simulation time exceeds the end time specified by the scene.
		 */
	template<typename Model, typename Limiter, typename ErrorEstimator>
	scalar_t Integrator<Model, Limiter, ErrorEstimator>::integrate(
	    auto substep_hook,
	    bool signal,
	    bool &force_break) {

		spdlog::debug("[Step {}]", n_global_steps);
		spdlog::debug("Global dt: {:.3e}", global_dt);
		spdlog::debug("Max. ACTI Level: {}", max_acti_level);

		global_step_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		/*
				Loops through ACTI levels in the correct order. e.g. for MAX_LEVEL = 3, the order is 2,2,1,2,2,1,0
			*/
		for (int acti_seq = 1; acti_seq <= (1 << (MAX_LEVEL - 1)); acti_seq++) {
			for (int l = 0; l < MAX_LEVEL; l++) {
				if (acti_seq % (1 << l) == 0) {
					ZoneScoped;

					int acti_level = MAX_LEVEL - l - 1;

					if (acti_clusters[acti_level].size() > 0) {
						if (acti_level > max_acti_level) {
							max_acti_level        = acti_level;
							acti_time[acti_level] = acti_time[acti_level - 1];
						}
					}

					if (acti_level > max_acti_level) continue;

					flux_update(acti_level);
					cell_update(acti_level, signal, force_break);

					if (force_break) return global_dt;

					slope_update(acti_level);

					if constexpr (Model::Multiphase) {
						smooth_alpha(acti_clusters[acti_level]);
						normal_update(acti_level);
						curvature_update(acti_level);
						smooth_curvature(acti_clusters[acti_level]);
					}
					adaptive_refinement(acti_level);

					acti_counter[acti_level]  = (acti_counter[acti_level] + 1) % 2;
					acti_time[acti_level]    += global_dt / (1 << acti_level);

					flush_acti_buffer(acti_level);

					n_substeps++;

					substep_hook(*this, acti_level, acti_time[acti_level], n_substeps, global_dt / (1 << acti_level));
				}
			}
		}

		scalar_t end_time  = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		scalar_t duration  = (end_time - global_step_start_time) / 1000.0;
		runtime_duration  += duration;

		int n_active_cells = 0;
		for (int ll = 0; ll < MAX_LEVEL; ll++) {
			n_active_cells += acti_clusters[ll].size();
		}

		scalar_t memory_efficiency = n_active_cells / (scalar_t) (domain.tree.size() - 1);
		n_global_steps++;

		if (memory_efficiency < 0.2) {
			spdlog::debug("Memory efficiency threshold met, garbage collecting...");
			gc();
		}

		return global_dt;
	}
}    // namespace dacti::integrator