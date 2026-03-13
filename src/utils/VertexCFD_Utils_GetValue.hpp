#ifndef VERTEXCFD_UTILS_GETVALUE_HPP
#define VERTEXCFD_UTILS_GETVALUE_HPP

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION double
getValue(const panzer::Traits::Residual::ScalarT& observed)
{
    return observed;
}

//---------------------------------------------------------------------------//
KOKKOS_INLINE_FUNCTION double
getValue(const panzer::Traits::Jacobian::ScalarT& observed)
{
    return observed.val();
}

//---------------------------------------------------------------------------//

} // namespace Utils
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_GETVALUE_HPP
