#ifndef VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONFACTORY_HPP
#define VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONFACTORY_HPP

#include "conduction/boundary_conditions/VertexCFD_BCStrategy_ConductionBoundaryFlux.hpp"

#include "utils/VertexCFD_Utils_BCStrategy_Defines.cpp"

#include <Panzer_BCStrategy_Factory.hpp>
#include <Panzer_BCStrategy_TemplateManager.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_Traits.hpp>

#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{

//---------------------------------------------------------------------------//
template<int NumSpaceDim>
class BCStractegyConductionFactory : public panzer::BCStrategyFactory
{
  public:
    Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits>>
    buildBCStrategy(
        const panzer::BC& bc,
        const Teuchos::RCP<panzer::GlobalData>& global_data) const override
    {
        // Check if BC strategy is `ConductionBoundaryFlux`
        if (bc.strategy() == "ConductionBoundaryFlux")
        {
            // Initialize BC template manager
            const auto bcs_tm = Teuchos::rcp(
                new panzer::BCStrategy_TemplateManager<panzer::Traits>{});

            const Impl::TemplateBuilder<ConductionBoundaryFlux, NumSpaceDim>
                builder(bc, global_data);

            bcs_tm->buildObjects(builder);
            return bcs_tm;
        }
        else
        {
            return Teuchos::null;
        }
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONFACTORY_HPP
