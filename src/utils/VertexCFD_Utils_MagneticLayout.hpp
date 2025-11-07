#ifndef VERTEXCFD_UTILS_MAGNETICLAYOUT_HPP
#define VERTEXCFD_UTILS_MAGNETICLAYOUT_HPP

#include <Phalanx_DataLayout.hpp>

namespace VertexCFD
{
namespace Utils
{
// Build phalanx DataLayout corresponding to vector magnetic field
Teuchos::RCP<PHX::DataLayout>
buildMagneticLayout(const Teuchos::RCP<const PHX::DataLayout>& scalar_layout,
                    int num_mag_dims);

// Build phalanx DataLayout corresponding to vector magnetic field gradient
Teuchos::RCP<PHX::DataLayout>
buildMagneticGradLayout(const Teuchos::RCP<const PHX::DataLayout>& vector_layout,
                        int num_mag_dims);
} // namespace Utils
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_MAGNETICLAYOUT_HPP
