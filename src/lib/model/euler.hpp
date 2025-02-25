#pragma once

#include "config/util.hpp"
#include "core/types.hpp"
#include "integrator/cell.hpp"

namespace dacti::model {

	/**
	 * @brief Defines the Euler equations for compressible gas dynamics.
	 * 
	 */
	template<const size_t D>
	struct euler {
		/*
			settings
		*/
		static constexpr bool Multiphase  = false;
		static constexpr bool HasSource   = false;
		static constexpr bool Viscosity   = false;
		static constexpr bool HLLC_solver = true;

		/*
			constants
		*/
		static constexpr size_t NumDims = D;
		static constexpr size_t NumEqns = NumDims + 2;
		static constexpr size_t NumFlux = NumEqns;
		static constexpr size_t NumParm = 0;
		static constexpr size_t NumGrad = 0;
		static constexpr size_t iu      = 1;
		static constexpr size_t ip      = NumEqns - 1;

		static constexpr scalar_t gamma = 1.4;

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
		using cell_t = integrator::_internal::Cell<euler<NumDims>>;
		using face_t = integrator::_internal::CellFace<NumDims>;


		static flux_t F(const prim_t p, const size_t dim) {
			flux_t F;
			size_t in   = iu + dim;    // normal velocity index
			scalar_t ke = 0.0;         // specific kinetic energy

			for (size_t i = 0; i < NumDims; ++i) {
				F[iu + i]  = p[0] * p[in] * p[iu + i];
				ke        += 0.5 * p[iu + i] * p[iu + i];
			}

			F[0]   = p[in] * p[0];
			F[in] += p[ip];
			F[ip]  = p[in] * (gamma / (gamma - 1) * p[ip] + ke * p[0]);

			return F;
		}


		static flux_t F(const prim_t p, scalar_t u_projected, const size_t dim) {
			flux_t F;
			size_t in   = iu + dim;    // normal velocity index
			scalar_t ke = 0.0;         // specific kinetic energy

			for (size_t i = 0; i < NumDims; ++i) {
				F[iu + i]  = p[0] * u_projected * p[iu + i];
				ke        += 0.5 * p[iu + i] * p[iu + i];
			}

			F[0]   = u_projected * p[0];
			F[in] += p[ip];
			F[ip]  = u_projected * (gamma / (gamma - 1) * p[ip] + ke * p[0]);

			return F;
		}


		static matA_t A(const prim_t p, const size_t dim) {
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
			A[ip][in] = p[0] * sound_speed_2(p[0], p[ip]);
			A[in][ip] = 1.0 / p[0];

			return A;
		}

		static void compute_rhs(auto &cell) {
			cell.rhs.fill(0.0);

			for (size_t i = 0; i < NumEqns; ++i) {
				for (size_t j = 0; j < NumDims; ++j) {
					matA_t A_ = A(cell.get_p(), j);
					for (size_t k = 0; k < NumEqns; ++k) {
						cell.rhs[i] += A_[i][k] * cell.get_slope()[j][k];
					}
				}
			}
		}

		static prim_t middle_state(const prim_t &pn, const prim_t &pp, int dim) {
			size_t in = iu + dim;    // normal velocity index
			prim_t pm;               // middle state

			// estimate wave speeds
			scalar_t c_n = sound_speed(pn[0], pn[ip]);
			scalar_t c_p = sound_speed(pp[0], pp[ip]);
			scalar_t u   = 0.5_ds * (pn[in] + pp[in]);
			scalar_t c   = 0.5_ds * (c_n + c_p);
			scalar_t Sn  = std::min(u - c, pn[in] - c_n);
			scalar_t Sp  = std::max(u + c, pp[in] + c_p);
			scalar_t Sm  = (pp[ip] - pn[ip] + pn[0] * pn[in] * (Sn - pn[in]) - pp[0] * pp[in] * (Sp - pp[in])) / (pn[0] * (Sn - pn[in]) - pp[0] * (Sp - pp[in]));

			// compute middle state
			if (Sn > 0)
				pm = pn;
			else if (Sp < 0)
				pm = pp;
			else if (Sm > 0 && Sn < 0) {
				pm[0]  = pn[0] * (Sn - pn[in]) / (Sn - Sm);
				pm[ip] = pn[ip] + pn[0] * (Sn - pn[in]) * (Sm - pn[in]);
				for (size_t i = 0; i < NumDims; ++i) {
					pm[iu + i] = pn[iu + i];
				}
				pm[in] = Sm;
			} else {
				pm[0]  = pp[0] * (Sp - pp[in]) / (Sp - Sm);
				pm[ip] = pp[ip] + pp[0] * (Sp - pp[in]) * (Sm - pp[in]);
				for (size_t i = 0; i < NumDims; ++i) {
					pm[iu + i] = pp[iu + i];
				}
				pm[in] = Sm;
			}

			return pm;
		}


		static flux_t flux(const auto &cn, const auto &cp, const face_t &face) {
			scalar_t dt_n = 0.5_ds * (face.t_1 + face.t_0) - cn.get_t();
			scalar_t dt_p = 0.5_ds * (face.t_1 + face.t_0) - cp.get_t();
			prim_t pn     = cn.get_p(face.center, cn.get_t(), face.dim);
			prim_t pp     = cp.get_p(face.center, cp.get_t(), face.dim);

			prim_t pm;    // middle state

			if constexpr (HLLC_solver) {
				// evolve boundary extrapolated value by delta_t/2
				pn -= dt_n * cn.get_rhs();
				pp -= dt_p * cp.get_rhs();
				pm  = middle_state(pn, pp, face.dim);
			} else {
				size_t in = iu + face.dim;    // normal velocity index
				// linearize around state tilde
				prim_t p_tilde   = 0.5_ds * (pn + pp);    // estimate primitive variables at the face
				scalar_t u_tilde = p_tilde[in];
				scalar_t a_tilde = sound_speed(p_tilde[0], p_tilde[ip]);

				// Lambda function to backtrace the characteristics
				auto back_trace = [&](scalar_t lambda) {
					if (lambda > 0) {
						return cn.get_p(face.center - dt_n * lambda * face.normal(), cn.get_t());
					} else if (lambda < 0) {
						return cp.get_p(face.center - dt_p * lambda * face.normal(), cp.get_t());
					} else {
						return p_tilde;
					}
				};
				prim_t p_1 = back_trace(u_tilde - a_tilde);    // backtrace characteristic 1
				prim_t p_2 = back_trace(u_tilde);              // backtrace characteristic 2
				prim_t p_3 = back_trace(u_tilde + a_tilde);    // backtrace characteristic 3

				// compute middle state
				pm[ip] = 0.5 * (p_3[ip] + p_1[ip]) + 0.5 * p_tilde[0] * a_tilde * (p_3[in] - p_1[in]);
				pm[0]  = (pm[ip] - p_2[ip]) / (a_tilde * a_tilde) + p_2[0];
				pm[in] = (pm[ip] - p_1[ip]) / (p_tilde[0] * a_tilde) + p_1[in];
				linalg::for_each_codim<NumDims>(face.dim, [&](int i) {
					pm[iu + i] = p_2[iu + i];
				});
			}

			return F(pm, face.dim) * face.area() * (face.t_1 - face.t_0);
		}


		static scalar_t max_delta_time(const auto &cell) {

			// convective limit
			scalar_t c = sound_speed(cell.get_p());
			vecn_t vel = cell.get_velocity();
			vecn_t dt;

			for (size_t i = 0; i < NumDims; ++i) {
				dt[i] = cell.get_size()[i] / (c + std::abs(vel[i]));
			}

			return dt.minCoeff();
		}


		static prim_t primitive_from_conserved(const conv_t &u) {
			return primitive_variables(u);
		}


		static conv_t conserved_from_primitive(const prim_t &p) {
			return conserved_variables(p);
		}

		/*
			End of the model interface, the integrator does not call any functions below this point directly.
		*/
		static scalar_t pressure(const conv_t &u) {
			scalar_t ke = 0;
			for (size_t i = 0; i < NumDims; ++i) {
				ke += 0.5 * u[iu + i] * u[iu + i];
			}
			return (gamma - 1) * (u[ip] - ke / u[0]);
		}

		static scalar_t sound_speed(const prim_t &p) {
			return std::sqrt(gamma * p[ip] / p[0]);
		}

		static scalar_t sound_speed(const scalar_t rho, const scalar_t p) {
			return std::sqrt(gamma * p / rho);
		}

		static scalar_t sound_speed_2(const scalar_t rho, const scalar_t p) {
			return gamma * p / rho;
		}

		static bool supersonic(const prim_t &p) {
			scalar_t u2 = 0;
			for (size_t i = 0; i < NumDims; ++i) {
				u2 += p[iu + i] * p[iu + i];
			}
			return u2 > sound_speed_2(p[0], p[ip]);
		}

		static scalar_t mach(const prim_t &p) {
			scalar_t u2 = 0;
			for (size_t i = 0; i < NumDims; ++i) {
				u2 += p[iu + i] * p[iu + i];
			}
			return std::sqrt(u2 / sound_speed_2(p[0], p[ip]));
		}

		static prim_t primitive_variables(const conv_t &u) {
			prim_t p;
			p[0] = u[0];
			for (size_t i = 0; i < NumDims; ++i) {
				p[iu + i] = u[iu + i] / u[0];
			}
			p[ip] = pressure(u);
			return p;
		}

		static conv_t conserved_variables(const prim_t &p) {
			conv_t u;
			scalar_t ke = 0;
			u[0]        = p[0];
			for (size_t i = 0; i < NumDims; ++i) {
				u[iu + i]  = p[0] * p[iu + i];
				ke        += 0.5 * p[iu + i] * p[iu + i];
			}
			u[ip] = p[ip] / (gamma - 1) + p[0] * ke;
			return u;
		}

		static void dict(toml::node_view<toml::node> m_view, prim_t &p_ref, vecn_t &vel_ref, std::optional<vec2_t> &rho_ref) {
			toml::table r_table = *m_view["reference_values"].as_table();

			scalar_t rho = config::get_entry<scalar_t>(r_table, "rho");
			scalar_t prs = config::get_entry<scalar_t>(r_table, "p");

			if (r_table["vel_magnt"]) {
				scalar_t vel_magnt = config::get_entry<scalar_t>(r_table, "vel_magnt");
				scalar_t vel_angle = config::get_entry<scalar_t>(r_table, "vel_angle", 0.0);

				vel_ref[0] = vel_magnt * std::cos(vel_angle * std::numbers::pi / 180);
				vel_ref[1] = vel_magnt * std::sin(vel_angle * std::numbers::pi / 180);
			} else if (r_table["Ma"]) {
				scalar_t mach      = config::get_entry<scalar_t>(r_table, "Ma");
				scalar_t vel_angle = config::get_entry<scalar_t>(r_table, "vel_angle", 0.0);

				scalar_t vel_magnt = mach * std::sqrt(1.4 * prs / rho);

				// TODO: implement arbitrary velocity direction and angle
				if constexpr (D == 2) {
					vel_ref[0] = vel_magnt * std::cos(vel_angle * std::numbers::pi / 180);
					vel_ref[1] = vel_magnt * std::sin(vel_angle * std::numbers::pi / 180);
				} else if constexpr (D == 3) {
					vel_ref[0] = -vel_magnt * std::cos(vel_angle * std::numbers::pi / 180);
					vel_ref[1] = 0.0;
					vel_ref[2] = vel_magnt * std::sin(vel_angle * std::numbers::pi / 180);
				}
			} else {
				vel_ref = config::get_entry<vecn_t>(r_table, "vel");
			}

			p_ref[0] = rho;
			for (int i = 0; i < D; i++) {
				p_ref[iu + i] = vel_ref[i];
			}
			p_ref[ip] = prs;
		}
	};

}    // namespace dacti::model