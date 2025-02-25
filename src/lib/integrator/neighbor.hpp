#pragma once
#include <array>

namespace dacti::integrator::_internal {
	template<typename T, const size_t NumDim>
	class Neighbor {
	public:
		static constexpr size_t NumNeighbors = 2 * NumDim;

		std::array<T, NumNeighbors> array;

		void fill(T t) {
			array.fill(t);
		}

		void enumerate(auto f) {
			for (int dim = 0; dim < NumDim; ++dim) {
				for (int dir = 0; dir < 1; ++dir) {
					f(dim, dir, array[dim * 2 + dir]);
				}
			}
		}

		void enumerate(auto f) const {
			for (int dim = 0; dim < NumDim; ++dim) {
				for (int dir = 0; dir < 1; ++dir) {
					f(dim, dir, array[dim * 2 + dir]);
				}
			}
		}

		void for_each(auto f) {
			for (size_t i = 0; i < NumNeighbors; ++i) {
				f(array[i]);
			}
		}

		template<typename U>
		Neighbor<U, NumDim> map(auto f) const {
			Neighbor<U, NumDim> result;
			for (size_t i = 0; i < NumNeighbors; ++i) {
				result.array[i] = f(array[i]);
			}
			return result;
		}

		T &get(int dim, int dir) {
			return array[dim * 2 + dir];
		}

		const T &get(int dim, int dir) const {
			return array[dim * 2 + dir];
		}

		Neighbor<T, NumDim> &operator=(const std::array<std::array<T, 2>, NumDim> &other) {

			for (int dim = 0; dim < NumDim; ++dim) {
				for (int dir = 0; dir < 2; ++dir) {
					array[dim * 2 + dir] = other[dim][dir];
				}
			}

			return *this;
		}


		operator std::array<T, NumNeighbors>() const {
			return array;
		}

		void clear() {
			for (int dim = 0; dim < NumDim; ++dim) {
				for (int dir = 0; dir < 2; ++dir) {
					array[dim * 2 + dir] = std::nullopt;
				}
			}
		}
	};
}    // namespace dacti::integrator::_internal