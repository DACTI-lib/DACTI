#include <optional>
#include <vector>

namespace util
{

    template<typename>
    struct is_array
    {
        static constexpr bool value()
        {
            return false;
        }
    };

    template<typename T, std::size_t N>
    struct is_array<std::array<T, N>>
    {
        static constexpr bool value()
        {
            return true;
        }
    };

    template<typename T>
    constexpr bool is_array_v = is_array<T>::value();


    template<typename T> requires is_array_v<T>
    struct array_size;

    template<typename T, const std::size_t N>
    struct array_size<std::array<T, N>>
    {
        constexpr static std::size_t value()
        {
            return N;
        }
    };

    template<typename T>
    constexpr std::size_t array_size_v = array_size<T>::value();

    template<typename T> requires is_array_v<T>
   struct array_element;

    template<typename T, const std::size_t N>
    struct array_element<std::array<T, N>>
    {
        using type = T;
    };

    template<typename T>
    using array_element_t = typename array_element<T>::type;

    template<bool ... b>
    constexpr bool all = requires()
    {
        (b,...);
    };

    template<typename... Ts>
    std::optional<std::size_t> common_size(const std::vector<Ts>&... cs)
    {

        if constexpr(sizeof...(Ts) == 0)
        {
            return 0;
        }

        std::array<std::size_t, sizeof...(Ts)> sizes = {cs.size()...};

        if constexpr(sizeof...(Ts) > 1)
        {
            for(int i = 0; i < sizeof...(Ts) - 1; i++)
            {
                if(sizes[i] != sizes[i+1])
                {
                    return std::nullopt;
                }
            }
        }

        return sizes[0];
    }




}
