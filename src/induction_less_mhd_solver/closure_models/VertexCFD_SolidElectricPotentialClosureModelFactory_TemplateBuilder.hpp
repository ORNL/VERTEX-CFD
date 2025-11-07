#ifndef VERTEXCFD_SOLIDELECTRICPOTENTIALCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_SOLIDELECTRICPOTENTIALCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_SolidElectricPotentialClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class SolidElectricPotentialFactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto solid_elec_pot = Teuchos::rcp(
            new SolidElectricPotentialFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            solid_elec_pot);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_SOLIDELECTRICPOTENTIALCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
