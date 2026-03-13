#ifndef VERTEXCFD_INCOMPRESSIBLELSVOFBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_INCOMPRESSIBLELSVOFBOUNDARYSTATE_FACTORY_HPP

#include "incompressible_lsvof_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleLSVOFFreeSlip.hpp"
#include "incompressible_lsvof_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleLSVOFNoSlip.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFBoundaryStateFactory
{
  public:
    static Teuchos::RCP<PHX::Evaluator<Traits>>
    create(const panzer::IntegrationRule& ir,
           const std::vector<std::string>& all_phase_names,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& lsvof_params,
           const Teuchos::ParameterList& /**user_params**/)
    {
        // Space dimension
        constexpr int num_space_dim = NumSpaceDim;

        // Number of LSVOF DOFs
        const int num_lsvof_dofs = all_phase_names.size() - 1;

        // LSVOF model type
        const std::string lsvof_model_name
            = lsvof_params.get<std::string>("LSVOF Model");

        // AC model type
        const std::string continuity_model_name
            = lsvof_params.isType<std::string>("Continuity Model")
                  ? lsvof_params.get<std::string>("Continuity Model")
                  : "AC";

        // Check if momentum equation should be built
        const bool build_mom_equ
            = lsvof_params.isType<bool>("Build LSVOF Navier-Stokes Equations")
                  ? lsvof_params.get<bool>(
                        "Build LSVOF Navier-Stokes Equations")
                  : true;

        // Loop over boundary conditions
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;

        if (bc_params.isType<std::string>("Type"))
        {
            if (bc_params.get<std::string>("Type") == "No-Slip")
            {
                state = Teuchos::rcp(
                    new IncompressibleLSVOFNoSlip<EvalType, Traits, num_space_dim>(
                        ir,
                        num_lsvof_dofs,
                        lsvof_model_name,
                        continuity_model_name,
                        build_mom_equ));

                found_model = true;
            }
            else if (bc_params.get<std::string>("Type") == "Free Slip")
            {
                state = Teuchos::rcp(
                    new IncompressibleLSVOFFreeSlip<EvalType, Traits, num_space_dim>(
                        ir,
                        num_lsvof_dofs,
                        lsvof_model_name,
                        continuity_model_name,
                        build_mom_equ));

                found_model = true;
            }
        }

        if (!found_model)
        {
            std::string msg = "\n\nBoundary state "
                              + bc_params.get<std::string>("Type")
                              + " failed to build.\n";
            msg += "The boundary conditions implemented in VertexCFD are:\n";
            msg += "Free Slip,\n";
            msg += "No-Slip,\n";
            msg += "\n";
            throw std::runtime_error(msg);
        }

        return state;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLELSVOFBOUNDARYSTATE_FACTORY_HPP
