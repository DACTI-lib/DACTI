#pragma once
#include <array>

namespace dacti::integrator::_internal {
	template<typename T, const size_t NumDim>
	class Children {
	public:
		static constexpr size_t NumChildren = 1 << NumDim;

		std::array<T, NumChildren> array;

		void fill(T t) {
			array.fill(t);
		}

		void enumerate(auto f) {
			auto arr = array;
			for (size_t i = 0; i < NumChildren; ++i) {
				f(i, arr[i]);
			}
		}

		void for_each(auto f) {
			for (size_t i = 0; i < NumChildren; ++i) {
				f(array[i]);
			}
		}

		void for_each(auto f) const {
			for (size_t i = 0; i < NumChildren; ++i) {
				f(array[i]);
			}
		}

		template<typename U>
		Children<U, NumDim> map(auto f) const {
			Children<U, NumDim> result;
			for (size_t i = 0; i < NumChildren; ++i) {
				result[i] = f(array[i]);
			}
			return result;
		}

		operator std::array<T, NumDim>() const { return array; }

		T &operator[](size_t i) { return array[i]; }
		const T &operator[](size_t i) const { return array[i]; }
	};
}    // namespace dacti::integrator::_internal
