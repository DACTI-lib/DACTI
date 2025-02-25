#pragma once

#include "integrator/cell.hpp"

namespace dacti::integrator::error {


	template<const size_t NumEqns>
	struct weighted_laplacian {

		static constexpr bool RequiresNeighborhood = true;

		vec_t<NumEqns> weights;

		weighted_laplacian(vec_t<NumEqns> weights) :
		    weights(weights) {}

		template<typename Model>
		scalar_t estimate(const _internal::Cell<Model> &cell, const auto &neighborhood) const {
			scalar_t error = 0.0;

			for (int dim = 0; dim < Model::NumDims; dim++) {
				auto d2udx2 = -2.0 * cell.p;

				for (int dir = 0; dir < 2; dir++) {
					if (!neighborhood.get(dim, dir)) {
						d2udx2 = Model::conv_t::Zero();
						break;
					}
					d2udx2 += *neighborhood.get(dim, dir);
				}

				error += d2udx2.cwiseMul(weights).squaredNorm();
			}

			return std::sqrt(error);
		}
	};

}    // namespace dacti::integrator::error