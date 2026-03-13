#ifndef VERTEXCFD_FULLINDUCTIONBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_FULLINDUCTIONBOUNDARYSTATE_FACTORY_HPP

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_FullInductionConducting.hpp"
#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_FullInductionFixed.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class FullInductionBoundaryStateFactory
{
  public:
    static constexpr int num_space_dim = NumSpaceDim;

    static Teuchos::RCP<PHX::Evaluator<Traits>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const MHDProperties::FullInductionMHDProperties& mhd_props)
    {
        // Loop over boundary conditions found in input file for the
        // induced magnetic field and scalar magnetic potential
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;

        if (bc_params.isType<std::string>("Type"))
        {
            const auto bc_type = bc_params.get<std::string>("Type");

            if (bc_type == "Conducting")
            {
                state = Teuchos::rcp(
                    new FullInductionConducting<EvalType, Traits, num_space_dim>(
                        ir, bc_params, mhd_props));
                found_model = true;
            }

            if (bc_type == "Fixed")
            {
                state = Teuchos::rcp(
                    new FullInductionFixed<EvalType, Traits, num_space_dim>(
                        ir, bc_params, mhd_props));
                found_model = true;
            }

            // Error message if model not found
            if (!found_model)
            {
                std::string msg = "\n\nBoundary state " + bc_type
                                  + " failed to build.\n";
                msg += "The boundary conditions implemented in VERTEX-CFD\n";
                msg += "for the full induction equations are:\n";
                msg += "Conducting,\n";
                msg += "Fixed,\n";
                msg += "\n";
                throw std::runtime_error(msg);
            }
        }

        // Return vector of evaluators
        return state;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONBOUNDARYSTATE_FACTORY_HPP
