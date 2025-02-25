#pragma once

#include "integrator/cell.hpp"
#include "model/common.hpp"
#include <cstddef>
#include <dacti.hpp>

namespace dacti::integrator {

	/**
	 *	@brief Define the type of a boundary face
	 */
	template<typename Model>
	struct Boundary {
	private:
		using prim_t = Model::prim_t;
		using flux_t = Model::flux_t;
		using vecn_t = Model::vecn_t;

		static constexpr size_t NumDims = Model::NumDims;

	public:
		enum class Type {
			zeroGradient,
			freeslip,
			noslip,
			inflow,
			inflow_r,
			outflow,
			dirichlet,
			neumann
		} type;

		prim_t p_in;
		vecn_t normal;
		scalar_t fade_in;
		/*
			Constructors used by the scene to set up boundary conditions
			return values are saved into scene::face_info
		*/
		static Boundary zeroGradient() {
			return {Type::zeroGradient, prim_t::Zero(), vecn_t::Zero(), 0.0};
		}

		static Boundary freeslip(vecn_t normal) {
			return {Type::freeslip, prim_t::Zero(), normal, 0.0};
		}

		static Boundary noslip(vecn_t normal) {
			return {Type::noslip, prim_t::Zero(), normal, 0.0};
		}

		static Boundary inflow(prim_t p_in, scalar_t fade_in) {
			return {Type::inflow, p_in, vecn_t::Zero(), fade_in};
		}

		static Boundary inflow_r(prim_t p_in, scalar_t fade_in) {
			return {Type::inflow_r, p_in, vecn_t::Zero(), fade_in};
		}

		static Boundary outflow(scalar_t pressure) {
			prim_t p_out     = prim_t::Zero();
			p_out[Model::ip] = pressure;
			return {Type::outflow, p_out, vecn_t::Zero(), 0.0};
		}

		static Boundary dirichlet(prim_t p_in) {
			return {Type::dirichlet, p_in, vecn_t::Zero(), 0.0};
		}

		/*
			return boundary values
			Only for fixed inlet conditions, member p remain unchanged; otherwise, p is updated based on boundary cell values
		*/
		prim_t boundary_value(const auto &cell) {

			prim_t p_cell = cell.get_p();

			if (type == Type::freeslip) {

				prim_t pb = p_cell;
				project_velocity(pb, normal);

				return pb;
			}

			if (type == Type::noslip) {
				prim_t pb = p_cell;
				for (size_t i = 0; i < NumDims; ++i) {
					pb[Model::iu + i] *= -1.0;
				}
				return pb;
			}

			if (type == Type::inflow || type == Type::inflow_r) {

				prim_t pb = p_in;

				if (!Model::supersonic(p_cell)) {
					pb[Model::ip] = p_cell[Model::ip];
				}

				if (fade_in > 0.0) {
					scalar_t vel_mult = std::min(cell.get_t() / fade_in, 1.0);
					for (int i = 0; i < NumDims; i++) {
						pb[Model::iu + i] *= vel_mult;
					}
				}

				return pb;
			}

			if (type == Type::dirichlet) {
				return p_in;
			}

			if (type == Type::outflow || type == Type::zeroGradient) {
				return p_cell;
			}

			assert(false && "Invalid boundary condition.");
			return prim_t::Zero();
		}

		// TODO: heat flux boundary condition not included yet

		/*
			compute boundary flux
		*/
		flux_t boundary_flux(const auto &cell, const _internal::CellFace<NumDims> &face) {
			flux_t flux;
			prim_t pb = boundary_value(cell);

			if (type == Type::freeslip) {

				flux = Model::F(cell.get_p(), pb[Model::iu + face.dim], face.dim);

			} else if (type == Type::noslip) {

				flux                       = flux_t::Zero();
				flux[Model::iu + face.dim] = pb[Model::ip];

			} else {
				prim_t pm;
				if constexpr (Model::Multiphase) {
					pm = Model::middle_state(pb, cell.get_p(), 0.0, face.dim);
				} else {
					pm = Model::middle_state(pb, cell.get_p(), face.dim);
				}
				flux = Model::F(pm, face.dim);
			}

			if constexpr (Model::Viscosity) {
				auto Fv = model::Fv<NumDims>(cell.get_gradient(), vec_t<NumDims>::Zero(), Model::mu(cell.get_p()), face.dim);
				Model::add_viscous_flux(Fv, flux);
			}


			return flux * face.area() * (face.t_1 - face.t_0) * face.direction();
		}

		/*
			for wall boundary conditions
		*/
		vecn_t vel_projected(const prim_t &p, const vecn_t &n) const {
			// extract velocity from primitive
			vecn_t vel;
			for (size_t i = 0; i < NumDims; ++i) {
				vel[i] = p[Model::iu + i];
			}

			// return projected velocity
			return vel - vel.dot(n) * n;
		}

		void project_velocity(prim_t &p, const vecn_t &n) {
			vecn_t vel = vel_projected(p, n);

			// assign it back to primitive
			for (size_t i = 0; i < NumDims; ++i) {
				p[Model::iu + i] = vel[i];
			}
		}
	};
};    // namespace dacti::integrator