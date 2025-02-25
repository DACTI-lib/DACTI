#pragma once

#include "core/types.hpp"

namespace dacti::model {


	template<const size_t D>
	vec_t<D + 1> Fv(const auto &du, const vec_t<D> &vel, const scalar_t mu, const scalar_t dim) {
		vec_t<D + 1> fv = vec_t<D + 1>::Zero();
		scalar_t trace  = 0.0;

		for (size_t i = 0; i < D; ++i) {
			fv[i]  = du[dim][i] + du[i][dim];
			trace += du[i][i];
		}
		fv[dim] -= (2.0 / 3.0 * trace);

		for (size_t i = 0; i < D; ++i) {
			fv[D] += fv[i] * vel[i];
		}

		return fv * mu;
	}
}    // namespace dacti::model