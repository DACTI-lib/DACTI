#pragma once

#include <dacti.hpp>

namespace dacti::integrator::slope {

	struct minmod {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const {
			if (s1 * s2 <= 0.0_ds)
				return 0.0_ds;
			else if (std::abs(s1) < std::abs(s2))
				return s1;
			else
				return s2;
		}
	};

	struct unlimited {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const {
			return 0.5 * (s1 + s2);
		}
	};

	struct van_leer {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const {
			scalar_t r = s1 / s2;
			return (r + std::abs(r)) / (1 + std::abs(r)) * s2;
		}
	};

	struct superbee {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const {
			scalar_t r = s1 / s2;
			return std::max(std::max(0.0_ds, std::min(2.0_ds * r, 1.0_ds)), std::min(r, 2.0_ds)) *
			       s2;
		}
	};

	struct von_alba_2 {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const {
			scalar_t r = s1 / s2;
			return 2.0_ds * r / (r * r + 1) * s2;
		}
	};

	struct koren {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const {
			scalar_t r = (s1 + 1e-9_ds) / (s2 + 1e-9_ds);
			return std::max(0.0_ds,
			                std::min(2.0 * r, std::min((1.0_ds + 2.0_ds * r) / 3.0_ds, 2._ds))) *
			       s2;
		}
	};

	struct first_order {
		inline scalar_t operator()(scalar_t s1, scalar_t s2) const { return 0.0_ds; }
	};
}    // namespace dacti::integrator::slope