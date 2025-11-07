#ifndef VERTEXCFD_UTILS_CONSTANTS_HPP
#define VERTEXCFD_UTILS_CONSTANTS_HPP

#include <Panzer_Traits.hpp>
#include <type_traits>

namespace VertexCFD
{
namespace Constants
{
// Generic pi for any floating point type.
template<typename T>
constexpr T pi_v = std::enable_if_t<std::is_floating_point<T>::value, T>{
    3.141592653589793238462643383279502884L};

// Most common case.
constexpr double pi = pi_v<double>;
constexpr unsigned int MAX_NUM_VIEW = 200;

} // namespace Constants
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_CONSTANTS_HPP
