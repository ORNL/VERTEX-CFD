#ifndef VERTEXCFD_ELECTRICPOTENTIALBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_ELECTRICPOTENTIALBOUNDARYSTATE_FACTORY_HPP

#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_ElectricCurrentDensityInsulating.hpp"
#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_ElectricPotentialFixed.hpp"
#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_ElectricPotentialInsulatingWall.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class ElectricPotentialBoundaryStateFactory
{
  public:
    static Teuchos::RCP<PHX::Evaluator<Traits>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& /*user_params*/)
    {
        constexpr int num_space_dim = NumSpaceDim;

        // Loop over boundary conditions
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;

        if (bc_params.isType<std::string>("Type"))
        {
            if (bc_params.get<std::string>("Type") == "Fixed")
            {
                state = Teuchos::rcp(
                    new ElectricPotentialFixed<EvalType, Traits>(ir, bc_params));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type")
                == "ElectricPotentialInsulatingWall")
            {
                state = Teuchos::rcp(
                    new ElectricPotentialInsulatingWall<EvalType, Traits>(ir));
                found_model = true;
            }

            if (bc_params.get<std::string>("Type")
                == "ElectricCurrentDensityInsulating")
            {
                state = Teuchos::rcp(
                    new ElectricCurrentDensityInsulating<EvalType,
                                                         Traits,
                                                         num_space_dim>(ir));
                found_model = true;
            }
        }

        if (!found_model)
        {
            std::string msg = "\n\nBoundary state "
                              + bc_params.get<std::string>("Type")
                              + " failed to build.\n";
            msg += "The boundary conditions implemented in VertexCFD\n";
            msg += "for the electric potential equation are:\n";
            msg += "Fixed,\n";
            msg += "ElectricCurrentDensityInsulating,\n";
            msg += "ElectricPotentialInsulatingWall,\n";
            msg += "\n";
            throw std::runtime_error(msg);
        }

        return state;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_ELECTRICPOTENTIALBOUNDARYSTATE_FACTORY_HPP
