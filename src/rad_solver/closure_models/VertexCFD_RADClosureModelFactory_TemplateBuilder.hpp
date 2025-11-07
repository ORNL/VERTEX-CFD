#ifndef VERTEXCFD_RADCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_RADCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_RADClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class RADFactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto rad_closure_factory
            = Teuchos::rcp(new RADFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            rad_closure_factory);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_RADCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
