#include "VertexCFD_Utils_PhaseLayout.hpp"
#include "VertexCFD_Utils_PhaseIndex.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_DataLayout_MDALayout.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace Utils
{
Teuchos::RCP<PHX::DataLayout>
buildPhaseLayout(const Teuchos::RCP<const PHX::DataLayout>& scalar_layout,
                 const int num_phases)
{
    return Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, PhaseIndex>(
            scalar_layout->extent(0), scalar_layout->extent(1), num_phases));
}

Teuchos::RCP<PHX::DataLayout>
buildPhaseGradLayout(const Teuchos::RCP<const PHX::DataLayout>& vector_layout,
                     const int num_phases)
{
    return Teuchos::rcp(
        new PHX::MDALayout<panzer::Cell, panzer::Point, panzer::Dim, PhaseIndex>(
            vector_layout->extent(0),
            vector_layout->extent(1),
            vector_layout->extent(2),
            num_phases));
}

} // namespace Utils
} // namespace VertexCFD
