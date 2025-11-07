#ifndef VERTEXCFD_INCOMPRESSIBLEBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_INCOMPRESSIBLEBOUNDARYSTATE_FACTORY_HPP

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleCavityLid.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleDirichlet.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleFreeSlip.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleLaminarFlow.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleNoSlip.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressiblePressureOutflow.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleRotatingWall.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleSymmetry.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleWallFunction.hpp"

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleBoundaryStateFactory
{
  public:
    static Teuchos::RCP<PHX::Evaluator<Traits>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& user_params,
           const FluidProperties::ConstantFluidProperties& fluid_prop)
    {
        // Space dimension
        constexpr int num_space_dim = NumSpaceDim;

        // EDAC vs AC
        const std::string continuity_model_name
            = user_params.isType<std::string>("Continuity Model")
                  ? user_params.get<std::string>("Continuity Model")
                  : "AC";

        const bool is_edac
            = continuity_model_name.find("EDAC") != std::string::npos ? true
                                                                      : false;

        // Loop over boundary conditions
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;

        if (bc_params.isType<std::string>("Type"))
        {
            if (bc_params.get<std::string>("Type") == "No-Slip")
            {
                state = Teuchos::rcp(
                    new IncompressibleNoSlip<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Dirichlet")
            {
                state = Teuchos::rcp(
                    new IncompressibleDirichlet<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Pressure Outflow")
            {
                state = Teuchos::rcp(
                    new IncompressiblePressureOutflow<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Free Slip")
            {
                state = Teuchos::rcp(
                    new IncompressibleFreeSlip<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Symmetry")
            {
                state = Teuchos::rcp(
                    new IncompressibleSymmetry<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Rotating Wall")
            {
                state = Teuchos::rcp(
                    new IncompressibleRotatingWall<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Laminar Flow")
            {
                state = Teuchos::rcp(
                    new IncompressibleLaminarFlow<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params, continuity_model_name));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Cavity Lid")
            {
                state = Teuchos::rcp(
                    new IncompressibleCavityLid<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, bc_params, is_edac));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type") == "Velocity Wall Function")
            {
                state = Teuchos::rcp(
                    new IncompressibleWallFunction<EvalType, Traits, num_space_dim>(
                        ir, fluid_prop, is_edac));
                found_model = true;
            }
        }

        if (!found_model)
        {
            std::string msg = "\n\nBoundary state "
                              + bc_params.get<std::string>("Type")
                              + " failed to build.\n";
            msg += "The boundary conditions implemented in VertexCFD are:\n";
            msg += "No-Slip,\n";
            msg += "Dirichlet,\n";
            msg += "Free Slip,\n";
            msg += "Pressure Outflow,\n";
            msg += "Rotating Wall,\n";
            msg += "Laminar Flow,\n";
            msg += "Symmetry,\n";
            msg += "Cavity Lid\n";
            msg += "Velocity Wall Function\n";
            msg += "\n";
            throw std::runtime_error(msg);
        }

        return state;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLEBOUNDARYSTATE_FACTORY_HPP
