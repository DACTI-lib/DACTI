#pragma once

#include "config/util.hpp"
#include "integrator/cell.hpp"
#include "model/common.hpp"

namespace dacti::model {

	/**
	 * @brief The `multi` model is based on the Allaire's Five-Equation-Model for compressible two-phase flows
	 * 
	 */
	template<const size_t D>
	struct multi {
		/*
			settings
		*/
		static constexpr bool Multiphase    = true;
		static constexpr bool HasSource     = true;
		static constexpr bool HLLC_solver   = true;
		static constexpr bool Viscosity     = true;
		static constexpr bool HeatDiffusion = true;

		/*
			constants
		*/
		static constexpr size_t NumDims = D;
		static constexpr size_t NumEqns = NumDims + 4;
		static constexpr size_t NumFlux = NumEqns + D + 1;
		static constexpr size_t NumParm = NumDims + 4;    // normal vector, curvature, viscosity, conductivity
		static constexpr size_t NumGrad = NumDims + 3;    // du, dv, dw, dT, dalpha, dbeta (beta is diffused alpha)
		// index in conserved/primitive variables
		static constexpr size_t iu = 2;              // the first velocity in primitive
		static constexpr size_t ip = NumEqns - 2;    // pressure
		static constexpr size_t ia = NumEqns - 1;    // alpha
		// index in parameter
		static constexpr size_t ikn    = 2;            // 1st component of interface normal
		static constexpr size_t kappa  = ikn + D;      // curvature
		static constexpr size_t alphas = kappa + 1;    // smoothed alpha field
		// index in gradient input
		static constexpr size_t dT = D;        // temperature gradient
		static constexpr size_t da = D + 1;    // alpha gradient

		/*
			parameters
		*/
		static constexpr scalar_t eps  = 1e-20;    // tolerance for flash calculation
		static constexpr scalar_t uc   = 1.0;      // compression velocity scale
		static constexpr scalar_t beta = 0.05;
		inline static scalar_t sigma   = 0.0;    // surface tension coefficient
		inline static scalar_t gamma1  = 6.12;
		inline static scalar_t gamma2  = 1.4;
		inline static scalar_t P1      = 343000000.0;
		inline static scalar_t P2      = 0.0;
		inline static scalar_t cp1     = 4181.0;
		inline static scalar_t cp2     = 1005.0;
		inline static scalar_t cv1     = cp1 / gamma1;
		inline static scalar_t cv2     = cp2 / gamma2;
		inline static scalar_t mu1     = 0.0;
		inline static scalar_t mu2     = 0.0;
		inline static scalar_t lambda1 = 0.0;
		inline static scalar_t lambda2 = 0.0;

		// for convenience
		using vecn_t = vec_t<NumDims>;
		using conv_t = vec_t<NumEqns>;
		using prim_t = vec_t<NumEqns>;
		using flux_t = vec_t<NumFlux>;
		using parm_t = vec_t<NumParm>;
		using grad_t = vec_t<NumGrad>;
		using matA_t = linalg::vec<prim_t, NumEqns>;
		using matS_t = linalg::vec<prim_t, NumDims>;
		using matG_t = linalg::vec<grad_t, NumDims>;
		using cell_t = integrator::_internal::Cell<multi<NumDims>>;
		using face_t = integrator::_internal::CellFace<NumDims>;


		static flux_t F(const prim_t p, const size_t dim) {
			flux_t F    = flux_t::Zero();
			size_t in   = iu + dim;    // normal velocity index
			scalar_t ke = 0.0;         // specific kinetic energy

			for (size_t i = 0; i < NumDims; ++i) {
				F[iu + i]  = (p[0] + p[1]) * p[in] * p[iu + i];
				ke        += 0.5 * p[iu + i] * p[iu + i];
			}

			F[0]   = p[in] * p[0];
			F[1]   = p[in] * p[1];
			F[in] += p[ip];
			F[ip]  = p[in] * (gamma(p[ia]) / (gamma(p[ia]) - 1) * (p[ip] + p0(p[ia])) + ke * (p[0] + p[1]));
			F[ia]  = p[in] * p[ia];

			F[ia + 1]       = p[in];
			F[ia + 2 + dim] = p[ia];

			return F;
		}

		/*
			projected flux
		*/
		static flux_t F(const prim_t p, scalar_t u_projected, const size_t dim) {
			flux_t F    = flux_t::Zero();
			size_t in   = iu + dim;    // normal velocity index
			scalar_t ke = 0.0;         // specific kinetic energy

			for (size_t i = 0; i < NumDims; ++i) {
				F[iu + i]  = (p[0] + p[1]) * u_projected * p[iu + i];
				ke        += 0.5 * p[iu + i] * p[iu + i];
			}

			F[0]   = u_projected * p[0];
			F[1]   = u_projected * p[1];
			F[in] += p[ip];
			F[ip]  = u_projected * (gamma(p[ia]) / (gamma(p[ia]) - 1) * (p[ip] + p0(p[ia])) + ke * (p[0] + p[1]));
			F[ia]  = u_projected * p[ia];

			F[ia + 1]       = u_projected;
			F[ia + 2 + dim] = p[ia];

			return F;
		}


		static matA_t A(const prim_t &p, const size_t ka, const size_t dim) {
			matA_t A;
			size_t in = iu + dim;    // normal velocity index

			for (size_t i = 0; i < NumEqns; ++i) {
				for (size_t j = 0; j < NumEqns; ++j) {
					if (i == j) A[i][j] = p[in];
					else
						A[i][j] = 0.0;
				}
			}

			A[0][in]  = p[0];
			A[1][in]  = p[1];
			A[ip][in] = (p[0] + p[1]) * sound_speed_2({p[0], p[1]}, p[ip], p[ia]);
			A[in][ip] = 1.0 / (p[0] + p[1]);
			A[in][ia] = sigma * ka / (p[0] + p[1]);

			return A;
		}


		static grad_t get_grad_input(const prim_t &p) {
			grad_t dp;
			for (size_t i = 0; i < NumDims; ++i) {
				dp[i] = p[iu + i];
			}
			dp[dT] = temperature({p[0], p[1]}, p[ip], p[ia]);
			dp[da] = p[ia];
			return dp;
		}

		static void compute_rhs(cell_t &cell) {
			cell.rhs.fill(0.0);

			for (size_t i = 0; i < NumEqns; ++i) {
				for (size_t j = 0; j < NumDims; ++j) {
					matA_t A_ = A(cell.get_p(), cell.get_k()[kappa], j);
					for (size_t k = 0; k < NumEqns; ++k) {
						cell.rhs[i] += A_[i][k] * cell.get_slope()[j][k];
					}
				}
			}
		}

		/**
		 * @brief Update parameters
		 */
		static void compute_parameters(cell_t &cell) {
			cell.k[0] = mu1 * cell.get_p()[ia] + mu2 * (1 - cell.get_p()[ia]);

			// diffused alpha field
			cell.k[alphas] = cell.get_p()[ia];
			// cell.k[alphas] = std::pow(cell.get_p()[ia], beta) / (std::pow(cell.get_p()[ia], beta) + std::pow(1 - cell.get_p()[ia], beta));

			// scalar_t _eps = 1e-6;
			// dp[da+1] = _eps * std::log((p[ia] + _eps) / (1.0 - p[ia] + _eps));
		}


		static conv_t source(const cell_t &cell, scalar_t dt, flux_t &du) {
			conv_t s = conv_t::Zero();

			for (size_t i = 0; i < NumDims; ++i) {
				s[iu + i] = -sigma * cell.get_k()[kappa] * du[ia + 2 + i];
			}
			s[ip] = sigma * cell.get_k()[kappa] * (-du[ia] + cell.get_p()[ia] * du[ia + 1]);
			s[ia] = -cell.get_p()[ia] * du[ia + 1];

			// for (size_t i = 0; i < NumDims; ++i) {
			// 	s[iu + i] = dt * sigma * cell.get_k()[kappa] * cell.get_gradient()[i][da]; // kappa * du/dx
			// 	s[ip] += s[iu + i] * cell.get_u()[iu + i];
			// }
			// s[ip] /= (cell.get_u()[0] + cell.get_u()[1]);
			// s[ia] = -cell.get_p()[ia] * du[ia+1];

			return s;
		}

		static prim_t middle_state(const prim_t &pn, const prim_t &pp, const scalar_t ka, int dim) {
			size_t in = iu + dim;

			// estimate wave speeds
			// scalar_t ka  = 0.0;
			scalar_t c_n = sound_speed({pn[0], pn[1]}, pn[ip], pn[ia]);
			scalar_t c_p = sound_speed({pp[0], pp[1]}, pp[ip], pp[ia]);
			scalar_t u   = 0.5_ds * (pn[in] + pp[in]);
			scalar_t c   = 0.5_ds * (c_n + c_p);
			scalar_t Sn  = std::min(u - c, pn[in] - c_n);
			scalar_t Sp  = std::max(u + c, pp[in] + c_p);
			scalar_t Sm  = (pp[ip] - pn[ip] + (pn[0] + pn[1]) * pn[in] * (Sn - pn[in]) - (pp[0] + pp[1]) * pp[in] * (Sp - pp[in]) - sigma * ka * (pp[ia] - pn[ia])) / ((pn[0] + pn[1]) * (Sn - pn[in]) - (pp[0] + pp[1]) * (Sp - pp[in]));
			// scalar_t Sm  = (pp[ip]-pn[ip] + (pn[0]+pn[1])*pn[in]*(Sn-pn[in]) - (pp[0]+pp[1])*pp[in]*(Sp-pp[in])) / ((pn[0]+pn[1])*(Sn-pn[in]) - (pp[0]+pp[1])*(Sp-pp[in]));

			// compute middle state
			prim_t pm;
			if (Sn > 0)
				pm = pn;
			else if (Sp < 0)
				pm = pp;
			else if (Sm > 0 && Sn < 0) {
				pm[0]  = pn[0] * (Sn - pn[in]) / (Sn - Sm);
				pm[1]  = pn[1] * (Sn - pn[in]) / (Sn - Sm);
				pm[ip] = pn[ip] + (pn[0] + pn[1]) * (Sn - pn[in]) * (Sm - pn[in]);
				pm[ia] = pn[ia];
				linalg::for_each_codim<NumDims>(dim, [&](size_t i) {
					pm[iu + i] = pn[iu + i];
				});
				pm[in] = Sm;
			} else {
				pm[0]  = pp[0] * (Sp - pp[in]) / (Sp - Sm);
				pm[1]  = pp[1] * (Sp - pp[in]) / (Sp - Sm);
				pm[ip] = pp[ip] + (pp[0] + pp[1]) * (Sp - pp[in]) * (Sm - pp[in]);
				pm[ia] = pp[ia];
				linalg::for_each_codim<NumDims>(dim, [&](size_t i) {
					pm[iu + i] = pp[iu + i];
				});
				pm[in] = Sm;
			}

			return pm;
		}


		static flux_t flux(const cell_t &cn, const cell_t &cp, const face_t &face) {
			size_t in = iu + face.dim;    // normal velocity index

			/*
				evolve boundary extrapolated value by delta_t/2
				using Eqs in primitive variables: Wt + A*Wx + B*Wy + C*Wz = 0
				W^{n+1} = W^n - dt/2 * (A*Wx + B*Wy + C*Wz)
			*/
			scalar_t dt_n = 0.5_ds * (face.t_1 + face.t_0) - cn.get_t();
			scalar_t dt_p = 0.5_ds * (face.t_1 + face.t_0) - cp.get_t();
			prim_t pn     = cn.get_p(face.center, cn.get_t(), face.dim) - dt_n * cn.get_rhs();
			prim_t pp     = cp.get_p(face.center, cp.get_t(), face.dim) - dt_p * cp.get_rhs();

			scalar_t ka = 0.5_ds * (cn.get_k()[kappa] + cp.get_k()[kappa]);
			prim_t pm   = middle_state(pn, pp, ka, face.dim);

			flux_t flux = F(pm, face.dim);

			/*
				artificial compression to sharpen the interface
			*/
			if (pm[ia] > 1e-5 && pm[ia] < 1 - 1e-5) {
				scalar_t n          = 0.5_ds * (cn.get_k()[ikn + face.dim] + cp.get_k()[ikn + face.dim]);
				scalar_t u_compress = uc * std::abs(pm[iu + face.dim]) * n;

				scalar_t ke = 0.0;
				for (size_t i = 0; i < NumDims; ++i) {
					flux[iu + i] += u_compress * ((1 - pm[ia]) * pm[0] - pm[ia] * pm[1]) * pm[iu + i];
					ke           += 0.5 * pm[iu + i] * pm[iu + i];
				}

				flux[0]  += u_compress * (1 - pm[ia]) * pm[0];
				flux[1]  += u_compress * (-pm[ia]) * pm[1];
				flux[ip] += u_compress * ((1 - pm[ia]) * pm[0] - pm[ia] * pm[1]) * ke +
				            u_compress * (1 - pm[ia]) * pm[ia] * (pm[ip] * (1 / (gamma1 - 1) - 1 / (gamma2 - 1)) + gamma1 * P1 / (gamma1 - 1) - gamma2 * P2 / (gamma2 - 1));
				flux[ia] += u_compress * (1 - pm[ia]) * pm[ia];
			}

			// viscous flux
			linalg::vec<grad_t, D> du;
			for (size_t i = 0; i < D; ++i) {
				du[i] = 0.5_ds * (cn.get_gradient()[i] + cp.get_gradient()[i]);
			}
			vecn_t vel = 0.5_ds * (cn.get_velocity() + cp.get_velocity());

			add_viscous_flux(Fv<D>(du, vel, mu(pm), face.dim), flux);

			// heat diffusion
			if constexpr (HeatDiffusion) {
				add_heat_flux(du[face.dim][dT] * lambda(pm), flux);
			}

			return flux * face.area() * (face.t_1 - face.t_0);
		}

		inline static void add_viscous_flux(const vec_t<D + 1> &flux_v, flux_t &flux) {
			for (size_t i = 0; i <= D; ++i)
				flux[iu + i] -= flux_v[i];
		}

		inline static void add_heat_flux(const scalar_t flux_heat, flux_t &flux) {
			flux[iu + D] -= flux_heat;
		}

		inline static scalar_t mu(const prim_t &p) {
			return mu1 * p[ia] + mu2 * (1 - p[ia]);
		}

		inline static scalar_t lambda(const prim_t &p) {
			return lambda1 * p[ia] + lambda2 * (1 - p[ia]);
		}


		static scalar_t max_delta_time(const auto &cell) {

			// convective limit
			scalar_t c = sound_speed(cell.get_p());
			vecn_t vel = cell.get_velocity();
			vecn_t dt;

			for (size_t i = 0; i < NumDims; ++i) {
				dt[i] = cell.get_size()[i] / (c + std::abs(vel[i]));
			}

			scalar_t dt_conv = dt.minCoeff();
			if constexpr (!Viscosity)
				return dt_conv;

			// viscous limit
			scalar_t k = std::max(mu(cell.get_p()), lambda(cell.get_p()) / std::min(cp1, cp2));

			if (k < 1e-15) {
				return dt_conv;
			} else {
				scalar_t dt_visc = cell.get_size()[0] * cell.get_size()[0] / k / 4.0;
				return std::min(dt_conv, dt_visc);
			}
		}

		/**
		 * @brief Converts a conserved variable vector to a primitive variable vector.
		 */
		static prim_t primitive_from_conserved(const conv_t &u) {
			return primitive_variables(u);
		}

		/**
		 * @brief Converts a primitive variable vector to a conserved variable vector.
		 */
		static conv_t conserved_from_primitive(const prim_t &p) {
			return conserved_variables(p);
		}


		/*
			Effective gamma
		 	1/(gamma-1) = alpha/(gamma_w-1) + (1-alpha)/(gamma_a-1)
		*/
		static scalar_t gamma(const scalar_t alpha) {
			if (alpha < eps)
				return gamma2;
			else if (alpha > 1 - eps)
				return gamma1;
			else
				return 1 + 1 / (alpha / (gamma1 - 1) + (1 - alpha) / (gamma2 - 1));
		}

		/*
			Effective p0
			gamma*p0/(gamma-1) = alpha*gamma_w*p0_w/(gamma_w-1)
		*/
		static scalar_t p0(const scalar_t alpha) {
			if (alpha < eps)
				return P2;
			else if (alpha > 1 - eps)
				return P1;
			else
				return (alpha * gamma1 * P1 / (gamma1 - 1) + (1 - alpha) * gamma2 * P2 / (gamma2 - 1)) / (1 + alpha / (gamma1 - 1) + (1 - alpha) / (gamma2 - 1));
		}

		/*
			Temperature
			rho*e  = rho*cv*T + p0
			rho*cv = alpha*rho_w*cv_w + (1-alpha)*rho_a*cv_a; p0 = alpha*p0_w + (1-alpha)*p0_a
		*/
		static scalar_t temperature(const vec2_t r, const scalar_t p, const scalar_t alpha) {
			if (alpha < eps)
				return (p + P2) / ((gamma2 - 1) * r[1] * cv2);
			else if (alpha > 1 - eps)
				return (p + P1) / ((gamma1 - 1) * r[0] * cv1);
			else
				return (p / (gamma(alpha) - 1) + alpha * P1 / (gamma1 - 1) + (1 - alpha) * P2 / (gamma2 - 1)) / (r[0] * cv1 + r[1] * cv2);
		}

		/*	
			Pressure
			rho*e = (p + gamma*p0) / (gamma-1)
		*/
		static scalar_t pressure(const conv_t &u) {
			scalar_t ke = 0;
			for (size_t i = 0; i < NumDims; ++i) {
				ke += 0.5 * u[iu + i] * u[iu + i];
			}
			scalar_t press = (gamma(u[ia]) - 1) * (u[ip] - ke / (u[0] + u[1])) - gamma(u[ia]) * p0(u[ia]);
			return press;
		}

		/*
			Speed of sound
			c^2 = gamma*(p+p0)/rho
		*/
		static scalar_t sound_speed(const prim_t &p) {
			return std::sqrt(gamma(p[ia]) * (p[ip] + p0(p[ia])) / (p[0] + p[1]));
		}

		static scalar_t sound_speed(const vec2_t r, const scalar_t p, const scalar_t a) {
			return std::sqrt(gamma(a) * (p + p0(a)) / (r[0] + r[1]));
		}

		static scalar_t sound_speed_2(const vec2_t r, const scalar_t p, const scalar_t a) {
			return gamma(a) * (p + p0(a)) / (r[0] + r[1]);
		}

		static bool supersonic(const prim_t &p) {
			scalar_t u2 = 0;
			for (size_t i = 0; i < NumDims; ++i) {
				u2 += p[iu + i] * p[iu + i];
			}
			return u2 > sound_speed_2(p[0], p[ip], p[ia]);
		}

		static scalar_t mach(const prim_t &p) {
			scalar_t u2 = 0;
			for (size_t i = 0; i < NumDims; ++i) {
				u2 += p[iu + i] * p[iu + i];
			}
			return std::sqrt(u2 / sound_speed_2(p[0], p[ip], p[ia]));
		}

		static prim_t primitive_variables(const conv_t &u) {
			prim_t p;
			p[0] = u[0];
			p[1] = u[1];
			for (size_t i = 0; i < NumDims; ++i) {
				p[iu + i] = u[iu + i] / (u[0] + u[1]);
			}
			p[ip] = pressure(u);
			p[ia] = u[ia];
			return p;
		}

		static conv_t conserved_variables(const prim_t &p) {
			conv_t u;
			scalar_t ke = 0;
			u[0]        = p[0];
			u[1]        = p[1];
			for (size_t i = 0; i < NumDims; ++i) {
				u[iu + i]  = (p[0] + p[1]) * p[iu + i];
				ke        += 0.5 * p[iu + i] * p[iu + i];
			}
			u[ip] = (p[ip] + gamma(p[ia]) * p0(p[ia])) / (gamma(p[ia]) - 1) + (p[0] + p[1]) * ke;
			u[ia] = p[ia];
			return u;
		}

		static void dict(toml::node_view<toml::node> m_view, prim_t &p_ref, vecn_t &vel_ref, std::optional<vec2_t> &rho_ref) {
			toml::table mod_table = *m_view.as_table();
			toml::table ref_table = *m_view["reference_values"].as_table();

			// parse reference primitive variables
			vec2_t rho     = config::get_entry<vec2_t>(ref_table, "rho");
			scalar_t press = config::get_entry<scalar_t>(ref_table, "p");
			scalar_t alpha = config::get_entry<scalar_t>(ref_table, "alpha");

			rho_ref = rho;

			if (ref_table["vel_magnt"]) {
				scalar_t vel_magnt = config::get_entry<scalar_t>(ref_table, "vel_magnt");
				scalar_t vel_angle = config::get_entry<scalar_t>(ref_table, "vel_angle", 0.0);

				vel_ref[0] = vel_magnt * std::cos(vel_angle * std::numbers::pi / 180);
				vel_ref[1] = vel_magnt * std::sin(vel_angle * std::numbers::pi / 180);

			} else if (ref_table["Ma"]) {
				scalar_t mach      = config::get_entry<scalar_t>(ref_table, "Ma");
				scalar_t vel_angle = config::get_entry<scalar_t>(ref_table, "vel_angle", 0.0);
				vec2_t rho_a       = {rho[0] * alpha, rho[1] * (1 - alpha)};
				scalar_t vel_magnt = mach * sound_speed(rho_a, press, alpha);

				vel_ref[0] = vel_magnt * std::cos(vel_angle * std::numbers::pi / 180);
				vel_ref[1] = vel_magnt * std::sin(vel_angle * std::numbers::pi / 180);
			} else {
				vel_ref = config::get_entry<vecn_t>(ref_table, "vel");
			}

			p_ref[0] = rho[0] * alpha;
			p_ref[1] = rho[1] * (1.0 - alpha);
			for (int i = 0; i < NumDims; i++) {
				p_ref[iu + i] = vel_ref[i];
			}
			p_ref[ip] = press;
			p_ref[ia] = alpha;


			// parse fluid properties
			scalar_t L        = config::get_entry<scalar_t>(ref_table, "L", 1.0);
			scalar_t mu_ratio = config::get_entry<scalar_t>(ref_table, "mu_ratio", 1.0);
			scalar_t Re, We, Oh = 0.0;

			// surface tension
			if (ref_table["We"]) {
				We    = config::get_entry<scalar_t>(ref_table, "We");
				sigma = 1.0 / We;
			}

			// viscosity
			if (ref_table["Re"]) {
				Re  = config::get_entry<scalar_t>(ref_table, "Re");
				mu1 = std::sqrt(rho[0] * sigma * L) * std::sqrt(We) / Re;
				mu2 = mu1 / mu_ratio;
			}

			if (ref_table["Oh"]) {
				Oh    = config::get_entry<scalar_t>(ref_table, "Oh");
				sigma = config::get_entry<scalar_t>(ref_table, "sigma");
				mu1   = std::sqrt(rho[0] * sigma * L) * Oh;
				mu2   = mu1 / mu_ratio;
			}

			// thermo properties of fluid 1
			if (mod_table.contains("fluid_1")) {
				P1     = config::get_entry<scalar_t>(*mod_table["fluid_1"].as_table(), "P0", P1);
				gamma1 = config::get_entry<scalar_t>(*mod_table["fluid_1"].as_table(), "gamma", gamma1);
			}

			// thermo properties of fluid 2
			if (mod_table.contains("fluid_2")) {
				P2     = config::get_entry<scalar_t>(*mod_table["fluid_2"].as_table(), "P0", P2);
				gamma2 = config::get_entry<scalar_t>(*mod_table["fluid_2"].as_table(), "gamma", gamma2);
			}
		}
	};

}    // namespace dacti::model