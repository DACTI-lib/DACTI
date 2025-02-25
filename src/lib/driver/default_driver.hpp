#pragma once

#include "config/config.hpp"
#include "core/util.hpp"
#include <chrono>
#include <dacti.hpp>
#include <fenv.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>


namespace dacti::driver {

	template<typename Integrator, typename Model, typename Writer, typename... Observers>
	class default_driver {
	private:
		Integrator integrator;
		Writer writer;
		std::shared_ptr<config::Config<Model>> config;

		// from observers
		std::tuple<Observers...> observers;
		std::vector<std::vector<scalar_t>> all_FieldVars;
		std::vector<scalar_t> all_IntegVars;
		std::vector<std::string> all_FieldVar_names;
		std::vector<std::string> all_IntegVar_names;

	public:
		// driver constructor with analytic observer
		default_driver(
		    const Integrator &integrator,
		    const Writer &writer,
		    std::shared_ptr<config::Config<Model>> config,
		    std::tuple<Observers...> observers) :
		    integrator(integrator),
		    writer(writer),
		    config(config),
		    observers(observers) {}

		void observe_all_variables(std::ofstream &observables_csv,
		                           const bool init_observation  = true,
		                           const bool observe_FieldVars = true) {
			util::for_each_in_tuple(
			    [&](const auto &observer) {
				    if (init_observation) {
					    for (const auto &FieldVar_name: observer.FieldVar_names()) {
						    all_FieldVar_names.push_back(fmt::format("{}/{}", observer.name(), FieldVar_name));
					    }
					    for (const auto &IntegVar_name: observer.IntegVar_names()) {
						    all_IntegVar_names.push_back(IntegVar_name);
						    observables_csv << "," << IntegVar_name;
					    }
				    }

				    observer.observe(integrator, config, observe_FieldVars);

				    spdlog::info("[[{}]]", observer.name());

				    for (int i = 0; i < observer.NumIntegVars(); ++i) {
					    all_IntegVars.push_back(observer.IntegVars()[i]);
					    spdlog::info("{}: {}", observer.IntegVar_names()[i], observer.IntegVars()[i]);
					    if (!init_observation) observables_csv << "," << observer.IntegVars()[i];
				    }

				    if (observe_FieldVars) {
					    for (int i = 0; i < observer.NumFieldVars(); ++i) {
						    all_FieldVars.push_back(std::move(observer.FieldVars()[i]));
					    }
				    }
			    },
			    observers);
		}

		void run() {
			spdlog::set_level(spdlog::level::info);
			spdlog::info("Starting simulation \"" + config->case_name + "\"");
			spdlog::info("Number of threads: {}", omp_get_max_threads());

			auto start_time                  = std::chrono::steady_clock::now();
			std::optional<scalar_t> end_time = config->end_time;
			scalar_t print_interval          = config->io_interval;
			double time                      = 0.0;
			double last_print_time           = 0.0;
			int frame_index                  = 0;
			int step_index                   = 0;
			int last_print_step              = 0;
			bool converged                   = false;

			// create a csv file to store time evolution of integral variables
			// if there are no user-defined observables, the default quantities stored are
			// - time, frame, step, cells
			std::ofstream observables_csv = std::ofstream(
			    fmt::format("{}/{}_observables.csv", config->result_out_dir, config->case_name));
			observables_csv << "time,frame,step,n_cells";

			// write the initial frame
			observe_all_variables(observables_csv);
			observables_csv << "\n";
			writer.write_data(integrator, 0.0, 0, all_FieldVars, all_FieldVar_names);

			// write frame
			auto try_write = [&](auto &integrator,
			                     int acti_level,
			                     scalar_t acti_time,
			                     int n_substeps,
			                     scalar_t dt,
			                     bool force_break) {
				// Check convergence for steady state simulations
				if (config->convergence_tol > 0) {
					assert(all_IntegVar_names.size() > 0 && "Convergence check requires at least one observed variable");
					spdlog::info("[[Observables]]");

					observables_csv << time << "," << frame_index << "," << step_index << "," << integrator.n_active_cells();

					std::vector<scalar_t> all_IntegVars_old = all_IntegVars;
					all_IntegVars.clear();

					// Get observables
					observe_all_variables(observables_csv, false, false);
					observables_csv << "\n";
					observables_csv.flush();

					scalar_t max_delta = 0.0;
					for (int i = 0; i < all_IntegVars.size(); i++) {
						max_delta = std::max(max_delta, std::abs(all_IntegVars[i] - all_IntegVars_old[i]) / dt);
					}

					spdlog::info("Max Delta: {:.2e} (tol: {:.2e})", max_delta, config->convergence_tol);

					if (max_delta < config->convergence_tol) {
						spdlog::info("Converged!");
						converged = true;
					}
				}    // Check convergence for steady state simulations


				// Write data
				if (acti_time - last_print_time >= print_interval || converged || force_break) {
					frame_index++;

					fmt::print("\n");

					spdlog::info("[Case \"{}\" | Frame {}]", config->case_name, frame_index);
					if (acti_level > 0)
						spdlog::info("ACTI Level: {}", acti_level);
					spdlog::info("Integration Steps: {}", step_index - last_print_step);
					if (end_time) {
						spdlog::info("Sim. Time: {:.2e} / {:.2e} ({:.1f}%)", time, *end_time, 100.0 * time / *end_time);
					}
					spdlog::info("Number of Cells: {}", integrator.n_active_cells());
					spdlog::info("Global DT: {:.2e}", integrator.global_dt);

					double cpu_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time).count();
					spdlog::info("CPU Time: {:.2f}s", cpu_seconds);

					if (end_time) {
						double eta = (*end_time - time) * cpu_seconds / (time + 1e-12);
						if (acti_level < 0) spdlog::info("ETA: {:.2f}s", eta);
						// spdlog::info("Number of flux computations: {}", integrator.total_flux_computations);
						// spdlog::info("Number of cell updates: {}", integrator.total_cell_updates);
					}

					// Get observables and additional fields in the observer
					all_FieldVars.clear();
					observables_csv << time << "," << frame_index << "," << step_index << "," << integrator.n_active_cells();

					if (all_IntegVar_names.size() > 0)
						spdlog::info("[Observables]");

					observe_all_variables(observables_csv, false);

					observables_csv << "\n";
					observables_csv.flush();

					last_print_time = acti_time;
					last_print_step = step_index;

					// write results
					writer.write_data(integrator, time, frame_index, all_FieldVars, all_FieldVar_names);

#ifndef NDEBUG
					integrator.save_debug_info(frame_index);
#endif
					return true;
				}
				return false;
			};


			//feenableexcept(FE_INVALID);

			bool force_break = false;

			while ((!end_time || time < *end_time) && !converged) {
				auto substep_hook =
				    [](auto &integrator,
				       int acti_level,
				       scalar_t acti_time,
				       int n_substeps,
				       scalar_t dt) { return; };

				scalar_t global_dt  = integrator.integrate(substep_hook, time > config->signal_time, force_break);
				time               += global_dt;

				try_write(integrator, -1, time, 0, global_dt, force_break);

				if (force_break) {
					spdlog::error("Forcing break due to nan");
					break;
				}

				step_index++;
			}
		}
	};
}    // namespace dacti::driver