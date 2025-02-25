#pragma once

#include "core/types.hpp"
#include "integrator/boundary.hpp"
#include "integrator/cell.hpp"
#include "integrator/children.hpp"
#include "integrator/neighbor.hpp"

namespace dacti::integrator::_internal {

	enum class creation_op {
		SPLIT,
		UNSPLIT,
		INIT,
	};

	template<typename Model>
	class Node {
	private:
		static constexpr size_t NumDim = Model::NumDims;

	public:
		Cell<Model> cell;
		Model::flux_t flux;

		Children<size_t, NumDim> child_indices;
		Neighbor<size_t, NumDim> neighbors;
		Neighbor<std::optional<Boundary<Model>>, NumDim> boundaries;

		creation_op op;

		int acti_level;
		int tree_level;
		size_t parent_index;
		scalar_t error;
		scalar_t creation_time;
		scalar_t max_dt;
		scalar_t cfl;

		bool is_leaf;
		bool is_active;
		bool is_garbage;
		bool can_split;
		bool can_unsplit;
		bool should_split;
		bool is_boundary = false;

		std::optional<Cell<Model>> get_cell() const {
			if (is_leaf && !is_garbage && is_active) {
				return cell;
			} else {
				return std::nullopt;
			}
		}

		std::array<double, 4> get_metadata() const {
			return {(scalar_t) acti_level, error, max_dt, cfl};
		}

		size_t find_child(size_t child_index) const {
			for (int i = 0; i < 1 << NumDim; i++) {
				if (child_index == child_indices[i])
					return i;
			}
			return 0;
		}
	};
}    // namespace dacti::integrator::_internal