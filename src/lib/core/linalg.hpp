#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>

#include <Eigen/Dense>

namespace dacti::core::linalg {


	template<typename S>
	struct HasZero {
		static S get_zero();
	};

	template<typename S>
	S zero() {
		return HasZero<S>::get_zero();
	}

	template<typename S>
	struct HasOne {
		static S get_one();
	};

	template<typename S>
	S one() {
		return HasOne<S>::get_one();
	}


	template<typename S, const size_t N>
	class vec {
		std::array<S, N> m_data;



	public:
		static constexpr size_t Size = N;

		// constructor
		template<typename... Ss>
		constexpr vec(const Ss &...s) {
			m_data = std::array<S, N>{s...};
		}

		constexpr static vec unit(size_t index) {
			vec result = zero<vec>();
			result[index] = one<S>();
			return result;
		}

		constexpr static vec constant(S value) {
			vec result;

			for (size_t i = 0; i < N; i++) {
				result[i] = value;
			}

			return result;
		}

		constexpr static vec quadrant(size_t index) {

			vec result;

			for (size_t i = 0; i < N; i++) {
				if ((1 << i) & index) {
					result[i] = one<S>();
				} else {
					result[i] = -one<S>();
				}
			}

			return result;
		}

		[[nodiscard]] S &operator[](size_t index) {
			return m_data[index];
		}

		[[nodiscard]] const S &operator[](size_t index) const {
			return m_data[index];
		}

		[[nodiscard]] vec operator+(const vec &other_vec) const {
			vec result;

			for (size_t i = 0; i < N; i++) {
				result[i] = m_data[i] + other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec operator-(const vec &other_vec) const {
			vec result;

			for (size_t i = 0; i < N; i++) {
				result[i] = m_data[i] - other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec operator-() const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = -m_data[i];
			}

			return result;
		}

		template<typename S2>
		[[nodiscard]] vec operator*(const S2 &scalar) const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] * scalar;
			}

			return result;
		}

		vec &operator+=(const vec &other_vec) {
			*this = (*this + other_vec);
			return *this;
		}

		vec &operator-=(const vec &other_vec) {
			*this = (*this - other_vec);
			return *this;
		}

		template<typename S2>
		vec &operator*=(const S2 &s) {
			*this = (*this * s);
			return *this;
		}

		template<typename S2>
		vec &operator/=(const S2 &s) {
			*this = (*this / s);
			return *this;
		}

		template<size_t N2>
		vec<S, N2> project() const {
			auto result = zero<vec<S, N2>>();

			for (int i = 0; i < std::min(N, N2); i++) {
				result[i] = m_data[i];
			}

			return result;
		}

		template<size_t N2>
		vec<S, N2> project_with_default(S def) const {
			vec<S, N2> result = vec<S, N2>::constant(def);

			for (int i = 0; i < std::min(N, N2); i++) {
				result[i] = m_data[i];
			}

			return result;
		}

		vec wrap(const vec &n) {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = std::fmod(m_data[i], n[i]);
			}

			return result;
		}

		vec clamp(const vec &a, const vec& b) {
			vec result;

			for (int i = 0; i < N; i++) {
				if (m_data[i] < a[i]) {
					m_data[i] = a[i];
				} else if (m_data[i] > b[i]) {
					m_data[i] = b[i];
				}
			}
		}

		vec wrap(const vec &n_0, const vec &n_1) {
			vec delta = n_1 - n_0;

			vec result = *this;

			for(int i = 0; i < N; i++) {
				if(m_data[i] < n_0[i]) {
					result[i] += delta[i];
				} else if(m_data[i] > n_1[i]) {
					result[i] -= delta[i];
				}
			}

			return result;
		}

		template<typename S2>
		[[nodiscard]] vec operator/(const S2 &scalar) const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] / scalar;
			}

			return result;
		}

		[[nodiscard]] const S dot(const vec &other_vec) const {
			S result = zero<S>();

			for (int i = 0; i < N; i++) {
				result += m_data[i] * other_vec[i];
			}

			return result;
		}

		[[nodiscard]] bool operator==(const vec &other_vec) const {
			for (int i = 0; i < N; i++) {
				if (m_data[i] != other_vec[i]) return false;
			}
			return true;
		}

		[[nodiscard]] bool operator!=(const vec &other_vec) const {
			return !(*this == other_vec);
		}

		[[nodiscard]] const S &x() const {
			return m_data[0];
		}


		[[nodiscard]] const S &y() const {
			return m_data[1];
		}

		[[nodiscard]] const S &z() const {
			return m_data[2];
		}

		[[nodiscard]] S &x() {
			return m_data[0];
		}

		[[nodiscard]] S &y() {
			return m_data[1];
		}

		[[nodiscard]] S &z() {
			return m_data[2];
		}


		[[nodiscard]] static vec Zero() {
			return HasZero<vec>::get_zero();
		}

		[[nodiscard]] static vec One() {
			return HasOne<vec>::get_one();
		}

		[[nodiscard]] size_t size() {
			return N;
		}

		[[nodiscard]] S minCoeff() {
			S min = m_data[0];

			for (int i = 1; i < N; i++) {
				min = std::min(min, m_data[i]);
			}

			return min;
		}

		[[nodiscard]] S maxCoeff() const {
			S max = m_data[0];

			for (int i = 1; i < N; i++) {
				max = std::max(max, m_data[i]);
			}

			return max;
		}

		[[nodiscard]] vec cwiseMin(const vec &other_vec) {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = std::min(m_data[i], other_vec[i]);
			}

			return result;
		}

		[[nodiscard]] vec cwiseMax(const vec &other_vec) {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = std::max(m_data[i], other_vec[i]);
			}

			return result;
		}

		[[nodiscard]] vec cwiseMul(const vec &other_vec) const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] * other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec cwiseDiv(const vec &other_vec) const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] / other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec cwiseComp(const vec<bool, N> &other_vec) const {
			vec<bool, N> result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] == other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec<bool, N> cwiseGreaterThan(const vec<S, N> &other_vec) const {
			vec<bool, N> result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] > other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec<bool, N> cwiseLesserThan(const vec<S, N> &other_vec) const {
			vec<bool, N> result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] < other_vec[i];
			}

			return result;
		}

		[[nodiscard]] vec rotate(S angle) {
			vec result;

			result[0] = m_data[0] * std::cos(angle) - m_data[1] * std::sin(angle);
			result[1] = m_data[0] * std::sin(angle) + m_data[1] * std::cos(angle);

			return result;
		}

		[[nodiscard]] vec rotaten90() {
			return {m_data[1], -m_data[0]};
		}


		[[nodiscard]] vec cwiseSqrt() const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = std::sqrt(m_data[i]);
			}

			return result;
		}

		[[nodiscard]] vec cwiseAbs() const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = std::abs(m_data[i]);
			}

			return result;
		}

		[[nodiscard]] vec cwiseInv() const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = 1.0 / m_data[i];
			}

			return result;
		}

		[[nodiscard]] vec cwiseSign() const {
			vec result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i] > 0.0 ? 1.0 : m_data[i] < 0.0 ? -1.0
				                                                    : 0.0;
			}

			return result;
		}

		[[nodiscard]] S norm() const {
			return std::sqrt(this->dot(*this));
		}

		[[nodiscard]] S squaredNorm() const {
			return this->dot(*this);
		}

		[[nodiscard]] vec normalized() const {
			return *this / this->norm();
		}

		void fetch_add(const vec &other_vec) {
			for (int i = 0; i < N; i++) {
#pragma omp atomic
				m_data[i] += other_vec[i];
			}
		}

		[[nodiscard]] bool is_nan() const {
			for (int i = 0; i < N; i++) {
				if (std::isnan(m_data[i])) {
					return true;
				}
			}

			return false;
		}

		[[nodiscard]] S sum() const {
			S result = zero<S>();

			for (int i = 0; i < N; i++) {
				result += m_data[i];
			}

			return result;
		}

		[[nodiscard]] S product() const {
			S result = one<S>();

			for (int i = 0; i < N; i++) {
				result *= m_data[i];
			}

			return result;
		}

		template<typename OutS>
		[[nodiscard]] vec<OutS, N> unary_op(auto &&op) const {
			vec<OutS, N> result;

			for (int i = 0; i < N; i++) {
				result[i] = op(m_data[i]);
			}

			return result;
		}

		template<typename RhsS, typename OutS>
		[[nodiscard]] vec<OutS, N> binary_op(vec<RhsS, N> other, auto &&op) const {
			vec<OutS, N> result;

			for (int i = 0; i < N; i++) {
				result[i] = op(m_data[i], other[i]);
			}

			return result;
		}

		S pnorm(S n) const {
			S result = zero<S>();

			for (int i = 0; i < N; i++) {
				result += std::pow(std::abs(m_data[i]), n);
			}

			return std::pow(result, 1.0 / n);
		}

		[[nodiscard]] size_t cwise_op_bits(auto op) const {
			size_t index = 0;

			for(int i = 0; i < N; i++) {
				if(op(m_data[i])) {
					index |= 1 << i;
				}
			}

			return index;
		}

		void fill(S s) {
			for (int i = 0; i < N; i++) {
				m_data[i] = s;
			}
		}

		bool all() {
			bool result = true;

			for (int i = 0; i < N; i++) {
				result = m_data[i] && result;
			}

			return result;
		}

		static vec<S, N> from_iterable(const auto& v) {
			vec<S, N> result;

			for (int i = 0; i < N; i++) {
				result[i] = v[i];
			}

			return result;
		}

		template<typename T>
		T to_iterable() const {
			T result;

			for (int i = 0; i < N; i++) {
				result[i] = m_data[i];
			}

			return result;
		}

	};

	template<typename S>
	[[nodiscard]] S cross(const vec<S, 2> &v1, const vec<S, 2> &v2) {
		return v1[0] * v2[1] - v1[1] * v2[0];
	}

	template<typename S>
	[[nodiscard]] vec<S, 3> cross(const vec<S, 3> &v1, const vec<S, 3> &v2) {
		return {v1[1] * v2[2] - v1[2] * v2[1],
		        v1[2] * v2[0] - v1[0] * v2[2],
		        v1[0] * v2[1] - v1[1] * v2[0]};
	}

	template<typename S, const size_t N>
	[[nodiscard]] S dot(const vec<S, N> &v1, const vec<S, N> &v2) {
		return v1.dot(v2);
	}

	template<typename S, const size_t N, typename S2>
	[[nodiscard]] vec<S, N> operator*(const S2 &scalar, const vec<S, N> &vec) {
		return vec * scalar;
	}

	template<typename S, const size_t N>
	std::ostream &operator<<(std::ostream &out, const vec<S, N> v) {
		out << "[";
		for (int i = 0; i < N; i++) {
			out << v[i];

			if (i < N - 1) out << ", ";
		}
		out << "]";
		return out;
	}

	template<>
	struct HasZero<float> {
		static float get_zero() {
			return 0.0f;
		}
	};

	template<>
	struct HasZero<double> {
		static double get_zero() {
			return 0.0;
		}
	};
	template<>
	struct HasZero<int> {
		static int get_zero() {
			return 0;
		}
	};

	template<typename S, const size_t N>
	struct HasZero<vec<S, N>> {
		static vec<S, N> get_zero() {
			vec<S, N> v;

			for (int i = 0; i < N; i++) {
				v[i] = zero<S>();
			}

			return v;
		}
	};

	template<>
	struct HasOne<float> {
		static float get_one() {
			return 1.0f;
		}
	};

	template<>
	struct HasOne<double> {
		static double get_one() {
			return 1.0;
		}
	};

	template<>
	struct HasOne<int> {
		static int get_one() {
			return 1;
		}
	};

	template<typename S, const size_t N>
	struct HasOne<vec<S, N>> {
		static vec<S, N> get_one() {
			vec<S, N> v;

			for (int i = 0; i < N; i++) {
				v[i] = one<S>();
			}

			return v;
		}
	};

	template<const size_t N>
	inline void for_each_codim(int dim, auto func) {
		for (int i = 0; i < N; i++) {
			if (i != dim) {
				func(i);
			} 
		}
	}
}// namespace dacti::core::linalg
