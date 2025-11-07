#ifndef VERTEXCFD_UTILS_PHASELAYOUT_HPP
#define VERTEXCFD_UTILS_PHASELAYOUT_HPP

#include <Phalanx_DataLayout.hpp>

namespace VertexCFD
{
namespace Utils
{
// Build phalanx DataLayout corresponding to vector velocity field
Teuchos::RCP<PHX::DataLayout>
buildPhaseLayout(const Teuchos::RCP<const PHX::DataLayout>& scalar_layout,
                 const int num_phases);

// Build phalanx DataLayout corresponding to vector velocity gradient field
Teuchos::RCP<PHX::DataLayout>
buildPhaseGradLayout(const Teuchos::RCP<const PHX::DataLayout>& vector_layout,
                     const int num_phases);
} // namespace Utils
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_PHASELAYOUT_HPP
