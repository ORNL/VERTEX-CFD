#ifndef VERTEXCFD_CONDUCTIONBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_CONDUCTIONBOUNDARYSTATE_FACTORY_HPP

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "conduction/boundary_conditions/VertexCFD_BoundaryState_AdiabaticWall.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionThermalConductivity.hpp"

#include "boundary_conditions/VertexCFD_BoundaryState_VariableExtrapolate.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_VariableFixed.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class ConductionBoundaryStateFactory
{
  public:
    static constexpr int num_space_dim = NumSpaceDim;

    static std::vector<Teuchos::RCP<PHX::Evaluator<Traits>>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& closure_model)
    {
        // Evaluator vector to return
        std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>> evaluators;

        // Get material property. Note: it is assumed that the material
        // property is listed under 'Material Properties' in the closure model
        // of the input file.
        const auto mat_prop_list
            = closure_model.sublist("Material Properties");

        // Register thermal conductivity to be used on the boundary based on
        // the material property type set in the input file
        const auto closure_type = mat_prop_list.get<std::string>("Type");
        if (closure_type == "ConstantMaterialProperties")
        {
            const auto mat_prop = Teuchos::rcp(
                new ClosureModel::ConductionThermalConductivity<EvalType,
                                                                panzer::Traits>(
                    ir, mat_prop_list));
            evaluators.push_back(mat_prop);
        }

        // Loop over boundary conditions found in input file
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;
        if (bc_params.isType<std::string>("Type"))
        {
            const auto bc_type = bc_params.get<std::string>("Type");

            if (bc_type == "AdiabaticWall")
            {
                state = Teuchos::rcp(new AdiabaticWall<EvalType, Traits>(ir));
                evaluators.push_back(state);
                found_model = true;
            }

            if (bc_type == "Extrapolate")
            {
                state = Teuchos::rcp(new VariableExtrapolate<EvalType, Traits>(
                    ir, "temperature"));
                evaluators.push_back(state);
                found_model = true;
            }

            if (bc_type == "Fixed")
            {
                state = Teuchos::rcp(new VariableFixed<EvalType, Traits>(
                    ir, bc_params, "temperature"));
                evaluators.push_back(state);
                found_model = true;
            }

            // Error message if model not found
            if (!found_model)
            {
                std::string msg = "\n\nBoundary state " + bc_type
                                  + " failed to build.\n";
                msg += "The boundary conditions implemented in VERTEX-CFD\n";
                msg += "for the conduction equation are:\n";
                msg += "AdiabaticWall,\n";
                msg += "Extrapolate,\n";
                msg += "Fixed,\n";
                msg += "\n";
                throw std::runtime_error(msg);
            }
        }

        // Return evaluators
        return evaluators;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_CONDUCTIONBOUNDARYSTATE_FACTORY_HPP
