#ifndef VERTEXCFD_BOUNDARYCONDITION_FACTORY_HPP
#define VERTEXCFD_BOUNDARYCONDITION_FACTORY_HPP

#include "VertexCFD_BCStrategy_StrongDirichletMMS.hpp"
#include "conduction/boundary_conditions/VertexCFD_BCStrategy_ConductionBoundaryFlux.hpp"
#include "incompressible_lsvof_solver/boundary_conditions/VertexCFD_BCStrategy_IncompressibleLSVOFBoundaryFlux.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BCStrategy_IncompressibleBoundaryFlux.hpp"
#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BCStrategy_SolidElectricBoundaryFlux.hpp"
#include "utils/VertexCFD_Utils_BCStrategy_Defines.cpp"

#ifdef VERTEXCFD_ENABLE_FULL_INDUCTION_MHD
#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BCStrategy_FullInductionMHDBoundaryFlux.hpp"
#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BCStrategy_SolidFullInductionMHDBoundaryFlux.hpp"
#endif

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
class Factory : public panzer::BCStrategyFactory
{
  public:
    Teuchos::RCP<panzer::BCStrategy_TemplateManager<panzer::Traits>>
    buildBCStrategy(
        const panzer::BC& bc,
        const Teuchos::RCP<panzer::GlobalData>& global_data) const override
    {
        auto template_manager = Teuchos::rcp(
            new panzer::BCStrategy_TemplateManager<panzer::Traits>{});

        const auto& bc_strategy = bc.strategy();

        if (bc_strategy == "IncompressibleBoundaryFlux")
        {
            const Impl::TemplateBuilder<IncompressibleBoundaryFlux, NumSpaceDim>
                builder(bc, global_data);
            template_manager->buildObjects(builder);
        }
        else if (bc_strategy == "IncompressibleLSVOFBoundaryFlux")
        {
            const Impl::TemplateBuilder<IncompressibleLSVOFBoundaryFlux, NumSpaceDim>
                builder(bc, global_data);
            template_manager->buildObjects(builder);
        }
        else if (bc_strategy == "SolidInductionLessMHDBoundaryFlux")
        {
            const Impl::TemplateBuilder<SolidElectricBoundaryFlux, NumSpaceDim>
                builder(bc, global_data);
            template_manager->buildObjects(builder);
        }
        else if (bc_strategy == "StrongDirichletMMS")
        {
            const Impl::TemplateBuilder<StrongDirichletMMS> builder(
                bc, global_data);
            template_manager->buildObjects(builder);
        }
#ifdef VERTEXCFD_ENABLE_FULL_INDUCTION_MHD
        else if (bc_strategy == "FullInductionMHDBoundaryFlux")
        {
            const Impl::TemplateBuilder<FullInductionMHDBoundaryFlux, NumSpaceDim>
                builder(bc, global_data);
            template_manager->buildObjects(builder);
        }
        else if (bc_strategy == "SolidFullInductionMHDBoundaryFlux")
        {
            const Impl::TemplateBuilder<SolidFullInductionMHDBoundaryFlux,
                                        NumSpaceDim>
                builder(bc, global_data);
            template_manager->buildObjects(builder);
        }
#endif
        else
        {
            template_manager = Teuchos::null;
        }

        // Return the bcs template manager.
        return template_manager;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYCONDITION_FACTORY_HPP
