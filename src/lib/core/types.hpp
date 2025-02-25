#pragma once

#include "core/linalg.hpp"

using std::isfinite;
using std::isinf;
using std::isnan;
using namespace dacti::core;

namespace dacti::core {

	using scalar_t = double;

	constexpr scalar_t operator"" _ds(long double s) {
		return static_cast<scalar_t>(s);
	}

	constexpr scalar_t epsilon() {
		return std::numeric_limits<scalar_t>::epsilon();
	}

	template<const size_t D>
	using vec_t = linalg::vec<scalar_t, D>;

	using vec2_t = vec_t<2>;
	using vec3_t = vec_t<3>;
	using vec4_t = vec_t<4>;
	using vec5_t = vec_t<5>;
	using vec6_t = vec_t<6>;

	using vec2i_t = linalg::vec<int, 2>;
	using vec3i_t = linalg::vec<int, 3>;
	using vec4i_t = linalg::vec<int, 4>;

	enum dimension_t { DIM_X = 0,
		                 DIM_Y = 1 };

	enum class boundary_position { NEG_X,
		                             POS_X,
		                             NEG_Y,
		                             POS_Y,
		                             NEG_Z,
		                             POS_Z };

	template<typename T>
	struct is_vec {
		static constexpr bool value = false;
	};

	template<typename S, size_t N>
	struct is_vec<linalg::vec<S, N>> {
		static constexpr bool value = true;
	};

	template<typename T>
	constexpr bool is_vec_v = is_vec<T>::value;


}    // namespace dacti::core
