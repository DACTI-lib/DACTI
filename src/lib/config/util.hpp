#pragma once

#include <core/util.hpp>
#include <dacti.hpp>
#include <toml++/toml.hpp>

namespace dacti::config {

	inline std::string get_toml_typename(const toml::node_view<const toml::node> node) {
		std::stringstream ss;
		ss << node.type();
		return ss.str();
	}

	template<typename T>
	T get_entry(const toml::table &root, const std::string &path, std::optional<T> default_value = std::nullopt) {


		auto node = root.at_path(path);

		if (!node) {
			if (default_value) {
				return *default_value;
			} else {
				spdlog::error("Config: Couldn't find entry for {}.", path);
				throw std::runtime_error("Config Error");
			}
		}


		if constexpr (is_vec_v<T>) {
			if (!node.is_array()) {
				spdlog::error("Config: Type of {} should be array, but is {}", path, get_toml_typename(node));
				throw std::runtime_error("Config Error");
			}

			const auto arr = *node.as_array();

			if (arr.size() != T::Size) {
				spdlog::error("Config: Length of {} should be {}, but is {}", path, T::Size, arr.size());
				throw std::runtime_error("Config Error");
			}

			T v;

			for (int i = 0; i < T::Size; i++) {

				auto entry = toml::node_view{arr.get(i)};

				if (!entry.is_floating_point()) {
					spdlog::error("Config: Type of {}[{}] should be floating point, but is {}", path, i, get_toml_typename(entry));
					throw std::runtime_error("Config Error");
				}

				v[i] = entry.as_floating_point()->get();
			}

			return v;
		} else if constexpr (std::is_same_v<T, toml::table>) {
			if (!node.is_table()) {
				spdlog::error("Config: Type of {} should be table, but is {}", path, get_toml_typename(node));
				throw std::runtime_error("Config Error");
			}
		} else if constexpr (std::is_floating_point_v<T>) {
			if (!node.is_floating_point()) {
				spdlog::error("Config: Type of {} should be floating point, but is {}", path, get_toml_typename(node));
				throw std::runtime_error("Config Error");
			}

			return node.as_floating_point()->get();

		} else if constexpr (std::is_integral_v<T>) {
			if (!node.is_integer()) {
				spdlog::error("Config: Type of {} should be int, but is {}", path, get_toml_typename(node));
				throw std::runtime_error("Config Error");
			}

			int64_t value = node.as_integer()->get();

			if (std::is_unsigned_v<T> && node.as_integer()->get() < 0) {
				spdlog::error("Config: {} should be unsigned, but a value of {} was specified", path, value);
				throw std::runtime_error("Config Error");
			}

			return value;
		} else if constexpr (std::is_same_v<T, std::string>) {
			if (!node.is_string()) {
				spdlog::error("Config: Type of {} should be string, but is {}", path, get_toml_typename(node));
				throw std::runtime_error("Config Error");
			}

			return node.as_string()->get();
		} else if constexpr (std::is_same_v<T, bool>) {
			if (!node.is_boolean()) {
				spdlog::error("Config: Type of {} should be boolean, but is {}", path, get_toml_typename(node));
				throw std::runtime_error("Config Error");
			}

			return node.as_boolean()->get();
		} else {
			static_assert(false, "Unsupported type");
		}
	}

	template<typename S>
	inline S solve_eq(auto f, S x0, S x1) {
		const int maxit  = 100;
		const double eps = 1e-7;

		S f0 = f(x0);

		for (int i = 0; i < maxit; i++) {
			S f1 = f(x1);

			S x2 = x1 - f1 * (x1 - x0) / (f1 - f0);

			if (std::abs(x2 - x1) < eps) {
				return x2;
			}

			x0 = x1;
			f0 = f1;
			x1 = x2;
		}

		return x1;
	}

	struct flow_state {
		scalar_t rho;
		scalar_t u;
		scalar_t p;
	};

	inline flow_state compute_normal_shock(
	    flow_state pre_shock,
	    scalar_t shock_mach,
	    const scalar_t gamma = 1.4,
	    const scalar_t P0    = 0.0) {
		scalar_t r1 = pre_shock.rho;
		scalar_t p1 = pre_shock.p;
		scalar_t c1 = std::sqrt(gamma * (p1 + P0) / r1);

		scalar_t r_ratio = 1.0 - 2.0 / (gamma + 1) * (1 - 1 / (shock_mach * shock_mach));
		scalar_t r2      = r1 / r_ratio;
		scalar_t u2      = 2.0 / (gamma + 1) * (shock_mach - 1 / shock_mach) * c1;
		scalar_t p2      = p1 * (((2 * P0 / p1 + 1) * (1 - r_ratio) * gamma + 1 + r_ratio) / ((r_ratio - 1) * gamma + 1 + r_ratio));

		return flow_state{r2, u2, p2};
	}

	inline flow_state compute_oblique_shock(
	    flow_state inflow,
	    scalar_t deflection_angle,
	    const scalar_t gamma = 1.4,
	    const scalar_t P0    = 0.0) {
		scalar_t r1    = inflow.rho;
		scalar_t p1    = inflow.p;
		scalar_t Ma1   = inflow.u / std::sqrt(gamma * (p1 + P0) / r1);
		scalar_t theta = deflection_angle;

		auto beta_to_theta = [&](double beta) {
			return std::tan(theta) - 2 * (std::pow(Ma1 * std::sin(beta), 2) - 1) / (Ma1 * Ma1 * (gamma + std::cos(2 * beta)) + 2) / std::tan(beta);
		};

		scalar_t beta = solve_eq(beta_to_theta, 0.5 * theta, theta);

		scalar_t Ma1n = Ma1 * std::sin(beta);
		scalar_t r2   = r1 / (2 / (gamma + 1) * std::pow(Ma1n, -2) + (gamma - 1) / (gamma + 1));
		scalar_t p2   = p1 * (2 * gamma / (gamma + 1) * std::pow(Ma1n, 2) - (gamma - 1) / (gamma + 1));
		scalar_t Ma2n = std::sqrt((1 + (gamma - 1) / 2 * std::pow(Ma1n, 2)) / (gamma * std::pow(Ma1n, 2) - (gamma - 1) / 2));
		scalar_t Ma2  = Ma2n / std::sin(beta - theta);

		return flow_state{r2, Ma2, p2};
	}

}    // namespace dacti::config