#pragma once

#include "config/config.hpp"
#include "driver/observer/common.hpp"

namespace dacti::observer {

	class AeroForce : public Observer {
	private:
		/**
		TODO: The following io functionality should be standardized.
		TODO: Only works for 2D cases at the moment.
		**/
		mutable int step = 0;    //Current step of the simulation

		///@brief Boundary mesh along which to sample pressure, lazily initialized
		mutable bool mesh_initialized = false;
		mutable int n_samples         = -1;
		mutable std::vector<vec2_t> mesh_line_centers;    // centers of the line segments
		mutable std::vector<scalar_t> mesh_areas;         // area of the line segments
		mutable std::vector<vec2_t> mesh_normals;         // normals of the line segments

	public:
		AeroForce() :
		    Observer("Aerodynamic Forces", {"Cl", "Cd"}, {"Cp", "ObjectSurface"}) {}

		/// @brief Sample resolution for the pressure distribution, given in number of points per unit length
		void ensure_sample_points(const auto config, int sample_resolution = 10) const {
			if (mesh_initialized) return;

			n_samples = 0;

			for (auto obj: config->objects) {
				Eigen::MatrixXd V = obj.mesh.V;

				int N = V.rows();

				for (int i = 0; i < N; i += 2) {
					// TODO: Right now mesh periodicity/watertightness is assumed.
					vec2_t a{V(i, 0), V(i, 1)};
					vec2_t b{V((i + 2) % N, 0), V((i + 2) % N, 1)};

					// Constant for the line segment
					vec2_t ab       = b - a;
					scalar_t length = ab.norm();
					vec2_t normal   = (ab / length).rotaten90();


					int n_samples_local = std::max(static_cast<int>(sample_resolution * length), 1);

					for (int j = 0; j < n_samples_local; j++) {
						vec2_t sample_pos = a + (j + 0.5) * ab / n_samples_local;

						sample_pos -= normal * 0.01;    // move the sample point slightly off the surface

						mesh_line_centers.push_back(sample_pos);
						mesh_areas.push_back(length / n_samples_local);
						mesh_normals.push_back(normal);
					}

					n_samples += n_samples_local;
				}
			}

			mesh_initialized = true;
		}

		/**
		@brief Run the observer, computes the lift and drag coefficients and prints the pressure distribution to a file
		**/
		template<typename Integrator, typename Model>
		void observe(const Integrator &integrator,
		             const std::shared_ptr<config::Config<Model>> config,
		             const bool observe_FieldVars = true) const {
			ZoneScoped;

			//
			// observer integral variables
			//
			ensure_sample_points(config);

			scalar_t rho_ref = config->p_ref[0];
			scalar_t p_ref   = config->p_ref[Model::ip];
			auto vel_ref     = config->vel_ref;

			std::vector<scalar_t> Cp;

			// reciprocal of the reference kinetic energy
			scalar_t e_ref_r = 2.0 / (rho_ref * vel_ref.squaredNorm());

			for (int i = 0; i < n_samples; i++) {
				size_t cell_index = integrator.get_cell_index(mesh_line_centers[i]);

				Cp.push_back((integrator.get_node(cell_index).cell.p[Model::ip] - p_ref) * e_ref_r);
			}

			scalar_t Cl = 0.0, Cd = 0.0;

			for (int i = 0; i < n_samples; i++) {
				Cl += Cp[i] * mesh_areas[i] * mesh_normals[i][1];
				Cd += Cp[i] * mesh_areas[i] * mesh_normals[i][0];
			}

			step++;

			_IntegVars = {Cl, Cd};

			if (!observe_FieldVars) return;

			//
			// observer field variables
			//
			for (auto &_FieldVar: _FieldVars) {
				_FieldVar.clear();
				_FieldVar.resize(integrator.n_active_cells());
			}

			integrator.for_each_active_cell([&](int i, auto &node) {
				scalar_t C_p       = 0.0;
				bool ObjectSurface = false;

				if (node.is_boundary) {
					C_p           = (node.cell.p[Model::ip] - p_ref) / (0.5 * rho_ref * vel_ref.squaredNorm());
					ObjectSurface = true;
				}

				_FieldVars[0].push_back(C_p);
				_FieldVars[1].push_back(ObjectSurface);
			});
		}
	};
}    // namespace dacti::observer