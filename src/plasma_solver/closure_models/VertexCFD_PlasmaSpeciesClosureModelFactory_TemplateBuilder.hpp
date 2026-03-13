#ifndef VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
#define VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP

#include "VertexCFD_PlasmaSpeciesClosureModelFactory.hpp"

#include <Panzer_ClosureModel_Factory_Base.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class PlasmaSpeciesFactoryTemplateBuilder
{
  public:
    template<typename EvalT>
    Teuchos::RCP<panzer::ClosureModelFactoryBase> build() const
    {
        auto plasma_species_closure_factory
            = Teuchos::rcp(new PlasmaSpeciesFactory<EvalT, NumSpaceDim>{});
        return Teuchos::rcp_static_cast<panzer::ClosureModelFactoryBase>(
            plasma_species_closure_factory);
    }
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_TEMPLATEBUILDER_HPP
