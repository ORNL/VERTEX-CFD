#ifndef VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_FullInductionClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class FullInductionFactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto full_inudction_closure_factory
            = Teuchos::rcp(new FullInductionFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            full_inudction_closure_factory);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
