#pragma once

#include "core/types.hpp"
#include "model/common.hpp"
#include "model/euler.hpp"
namespace dacti::model {

	/**
	 * @struct navier_stokes, inherited from euler
	 * @brief The compressible Navier-Stokes equations
	 * 
	 */
	template<const size_t D>
	struct navier_stokes : public euler<D> {
		/*
			settings
		*/
		static constexpr bool Viscosity     = true;
		static constexpr bool HeatDiffusion = true;

		/*
			constants
		*/
		// static constexpr size_t NumParm = 2;			// viscosity(mu), conductivity(lambda)
		static constexpr size_t NumGrad = D + 1;    // du, dv, dw, dT

		static constexpr scalar_t cp = 1005.0;    // specific heat capacity at constant pressure
		static constexpr scalar_t R  = 287.0;     // gas constant

		inline static scalar_t mu_     = 0.0;
		inline static scalar_t lambda_ = 0.0;

		/*
			for convenience
		*/
		static constexpr size_t iu = euler<D>::iu;
		static constexpr size_t ip = euler<D>::ip;
		static constexpr size_t dT = D;


		using vecn_t = euler<D>::vecn_t;
		using conv_t = euler<D>::conv_t;
		using prim_t = euler<D>::prim_t;
		using flux_t = euler<D>::flux_t;
		using grad_t = vec_t<NumGrad>;
		using matG_t = linalg::vec<grad_t, D>;
		using cell_t = integrator::_internal::Cell<navier_stokes<D>>;
		using face_t = integrator::_internal::CellFace<D>;


		static grad_t get_grad_input(const prim_t &p) {
			grad_t dp;
			for (size_t i = 0; i < D; ++i) {
				dp[i] = p[iu + i];
			}
			dp[dT] = temperature(p[0], p[ip]);

			return dp;
		}

		static flux_t flux(const cell_t &cn, const cell_t &cp, const face_t &face) {
			// convective flux
			flux_t flux = euler<D>::flux(cn, cp, face);

			// viscous flux
			linalg::vec<grad_t, D> du;
			for (size_t i = 0; i < D; ++i) {
				du[i] = 0.5_ds * (cn.get_gradient()[i] + cp.get_gradient()[i]);
			}
			vecn_t vel = 0.5_ds * (cn.get_velocity() + cp.get_velocity());

			vec_t<D + 1> flux_v = Fv<D>(du, vel, mu(cn.get_p()), face.dim) * face.area() * (face.t_1 - face.t_0);

			add_viscous_flux(flux_v, flux);

			// heat diffusion
			if constexpr (HeatDiffusion) {
				scalar_t flux_heat = lambda_ * du[face.dim][dT] * face.area() * (face.t_1 - face.t_0);
				add_heat_flux(flux_heat, flux);
			}
			return flux;
		}

		inline static void add_viscous_flux(const vec_t<D + 1> &flux_v, flux_t &flux) {
			for (size_t i = 0; i <= D; ++i)
				flux[iu + i] -= flux_v[i];
		}

		inline static void add_heat_flux(const scalar_t flux_heat, flux_t &flux) {
			flux[iu + D] -= flux_heat;
		}

		inline static scalar_t mu(const prim_t &p) {
			return mu_;
		}

		inline static scalar_t lambda(const prim_t &p) {
			return lambda_;
		}

		static scalar_t max_delta_time(const cell_t &cell) {
			// convective limit
			scalar_t dt_conv = euler<D>::max_delta_time(cell);

			// viscous limit
			scalar_t k = std::max(mu(cell.get_p()), lambda(cell.get_p()) / cp);

			if (k < 1e-15) {
				return dt_conv;
			} else {
				scalar_t dt_visc = cell.get_size()[0] * cell.get_size()[0] / k * 0.5;
				return std::min(dt_conv, dt_visc);
			}
		}

		inline static scalar_t temperature(const scalar_t rho, const scalar_t p) {
			return p / (rho * R);
		}

		static void dict(toml::node_view<toml::node> m_view, prim_t &p_ref, vecn_t &vel_ref, std::optional<vec2_t> &rho_ref) {
			euler<D>::dict(m_view, p_ref, vel_ref, rho_ref);

			toml::table ref_table = *m_view["reference_values"].as_table();

			scalar_t Re = config::get_entry<scalar_t>(ref_table, "Re");
			scalar_t L  = config::get_entry<scalar_t>(ref_table, "L");

			mu_ = p_ref[0] * vel_ref.norm() * L / Re;

			if (ref_table["Pr"]) {
				scalar_t Pr = config::get_entry<scalar_t>(ref_table, "Pr");
				lambda_     = mu_ * cp / Pr;
			}
		}
	};

}    // namespace dacti::model