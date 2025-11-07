#ifndef VERTEXCFD_SOLIDFULLINDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_SOLIDFULLINDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_SolidFullInductionClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class SolidFullInductionFactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto solid_induction = Teuchos::rcp(
            new SolidFullInductionFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            solid_induction);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_SOLIDFULLINDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
