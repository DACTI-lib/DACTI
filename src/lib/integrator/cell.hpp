#pragma once

#include <cstddef>
#include <dacti.hpp>

namespace dacti::integrator::_internal {

	template<typename Model>
	class Cell {
	public:
		scalar_t size;
		scalar_t t;
		Model::vecn_t center;
		Model::conv_t u;
		Model::prim_t p;
		Model::conv_t rhs;
		Model::parm_t k;
		Model::grad_t t_slope;
		Model::matS_t slope;
		Model::matG_t gradient;


		Model::prim_t get_p(const Model::vecn_t &x, scalar_t t) const {
			auto ret = p;

			for (size_t i = 0; i < Model::NumDims; ++i) {
				ret += slope[i] * (x[i] - center[i]);
			}

			return ret;
		}

		Model::prim_t get_p(const Model::vecn_t &x, scalar_t t, size_t dim) const {
			auto ret  = p;
			ret      += slope[dim] * (x[dim] - center[dim]);

			return ret;
		}

		Model::conv_t get_u() const {
			return u;
		}

		Model::prim_t get_p() const {
			return p;
		}

		Model::prim_t get_slope(size_t dim) const {
			return slope[dim];
		}

		Model::parm_t get_k(const Model::vecn_t &x, scalar_t t) const {
			return k;
		}

		Model::parm_t get_k() const {
			return k;
		}

		Model::vecn_t get_velocity() const {
			typename Model::vecn_t vel;
			for (size_t i = 0; i < Model::NumDims; ++i) {
				vel[i] = p[Model::iu + i];
			}
			return vel;
		}

		Model::matS_t get_slope() const {
			return slope;
		}

		Model::matG_t get_gradient() const {
			return gradient;
		}

		Model::vecn_t get_center() const {
			return center;
		}

		Model::vecn_t get_size() const {
			return Model::vecn_t::constant(size);
		}

		scalar_t get_t() const {
			return t;
		}

		scalar_t volume() const {
			return std::pow(size, Model::NumDims);
		}

		Cell translated(vec2_t dx) const {
			auto copy    = *this;
			copy.center += dx;
			return copy;
		}

		Model::conv_t get_rhs() const {
			return rhs;
		}

		Model::vecn_t face_position(int dim, int dir) const {
			auto ret  = center;
			ret[dim] += (2 * dir - 1) * size / 2.0;
			return ret;
		}
	};


	template<const size_t NumDims>
	struct CellFace {
		vec_t<NumDims> center;
		scalar_t side_length;
		scalar_t t_0;
		scalar_t t_1;
		int dim;
		int dir;

		CellFace(vec_t<NumDims> center, scalar_t side_length, scalar_t t_0, scalar_t t_1, int dim, int dir) :
		    center(center),
		    side_length(side_length),
		    t_0(t_0),
		    t_1(t_1),
		    dim(dim),
		    dir(dir) {}

		scalar_t direction() const {
			return 2 * dir - 1;
		}

		vec_t<NumDims> normal() const {
			return vec_t<NumDims>::unit(dim) * direction();
		}

		scalar_t area() const {
			return std::pow(side_length, NumDims - 1);
		}
	};

}    // namespace dacti::integrator::_internal