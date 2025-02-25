#pragma once

#include <tuple>

namespace dacti::util {
	template <typename Func, typename Tuple, std::size_t... I>
    void for_each_in_tuple(Func&& func, Tuple&& tuple, std::index_sequence<I...>) {
        (func(std::get<I>(tuple)), ...);
    }

    template <typename Func, typename... Args>
    void for_each_in_tuple(Func&& func, const std::tuple<Args...>& tuple) {
        for_each_in_tuple(std::forward<Func>(func), tuple, std::index_sequence_for<Args...>{});
	}
}