#ifndef VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_ConductionClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class ConductionFactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto conduction_closure_factory
            = Teuchos::rcp(new ConductionFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            conduction_closure_factory);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
