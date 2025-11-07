#include "VertexCFD_Utils_MagneticLayout.hpp"
#include "VertexCFD_Utils_MagneticDim.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_DataLayout_MDALayout.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace Utils
{
Teuchos::RCP<PHX::DataLayout>
buildMagneticLayout(const Teuchos::RCP<const PHX::DataLayout>& scalar_layout,
                    int num_mag_dims)
{
    return Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, MagneticDim>(
            scalar_layout->extent(0), scalar_layout->extent(1), num_mag_dims));
}

Teuchos::RCP<PHX::DataLayout> buildMagneticGradLayout(
    const Teuchos::RCP<const PHX::DataLayout>& vector_layout, int num_mag_dims)
{
    return Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, panzer::Dim, MagneticDim>(
            vector_layout->extent(0),
            vector_layout->extent(1),
            vector_layout->extent(2),
            num_mag_dims));
}

} // namespace Utils
} // namespace VertexCFD
