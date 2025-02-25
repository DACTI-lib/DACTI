#pragma once

#include "integrator/integrator.hpp"

namespace dacti::integrator {
	/**
		 * @brief Assigns given nodes to ACTI clusters, ignoring nodes that are not leaves or not active.
		 * 
		 */
	template<typename M, typename L, typename E>
	void Integrator<M, L, E>::assign_to_acti_clusters(std::span<size_t> node_indices) {
		for (int i = 0; i < node_indices.size(); ++i) {
			auto &node = domain.tree[node_indices[i]];

			if (!node.is_active || !node.is_leaf) continue;

			node.cfl         = global_dt / node.max_dt;
			node.acti_level  = std::clamp((int) std::log2(node.cfl / max_cfl), 0, MAX_LEVEL - 1);
			node.cfl        /= (1 << node.acti_level);

			// Add leaf to ACTI cluster corresponding to the computed ACTI level.
			acti_clusters[node.acti_level].push_back(node_indices[i]);
		}
	}

	/**
		 * @brief Clears all ACTI clusters.
		 * 
		 */
	template<typename M, typename L, typename E>
	void Integrator<M, L, E>::clear_acti_clusters() {
		for (int i = 0; i < acti_clusters.size(); ++i) {
			acti_clusters[i].clear();
		}
	}

	/**
		 * @brief Adds deferred nodes from specified acti_level to the corresponding ACTI cluster.
		 * 
		 * @param acti_level ACTI buffer level to flush.
		 *
		 */
	template<typename M, typename L, typename E>
	void Integrator<M, L, E>::flush_acti_buffer(int acti_level) {
		for (size_t index: acti_buffers[acti_level]) {
			acti_clusters[acti_level].push_back(index);
			domain.tree[index].acti_level = acti_level;
		}
		acti_buffers[acti_level].clear();
	}

	/**
		 * @brief Garbage collects the domain.
		 * 
		 * This function loops through all nodes in the domain and removes all nodes that are not leaves and have no active children.
		 * It then reassigns all leaves to ACTI clusters.
		 */
	template<typename M, typename L, typename E>
	void Integrator<M, L, E>::gc() {
		ZoneScoped;
		domain.gc();
		std::vector<size_t> leaves;

		for (int i = 1; i < domain.tree.size(); ++i) {
			auto &node = domain.tree[i];

			if (node.is_leaf && node.is_active) {
				leaves.push_back(i);
			}
		}

		clear_acti_clusters();
		assign_to_acti_clusters(leaves);
	}
}    // namespace dacti::integrator