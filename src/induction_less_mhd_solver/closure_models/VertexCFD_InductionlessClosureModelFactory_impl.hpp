#ifndef VERTEXCFD_INDUCTIONLESSCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_INDUCTIONLESSCLOSUREMODELFACTORY_IMPL_HPP

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricCurrentDensity.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialCrossProductFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialDiffusionFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_HartmannProblemExact.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_JouleHeatingSource.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_LorentzForce.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_InductionlessClosureModelFactory.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void InductionlessFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const std::string& closure_type,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& user_params,
    const Teuchos::ParameterList& closure_params,
    bool& found_model,
    std::string& error_msg,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Define local variables
    constexpr int num_space_dim = NumSpaceDim;
    const Teuchos::RCP<PHX::Evaluator<panzer::Traits>> eval;

    // Closure models
    if (closure_type == "ElectricCurrentDensity")
    {
        auto eval = Teuchos::rcp(
            new ElectricCurrentDensity<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "ElectricPotentialCrossProductFlux")
    {
        auto eval = Teuchos::rcp(
            new ElectricPotentialCrossProductFlux<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(*ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "ElectricPotentialDiffusionFlux")
    {
        auto eval = Teuchos::rcp(
            new ElectricPotentialDiffusionFlux<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "HartmannProblemExact")
    {
        auto eval = Teuchos::rcp(
            new HartmannProblemExact<EvalType, panzer::Traits, num_space_dim>(
                *ir, closure_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "JouleHeatingSource")
    {
        auto eval = Teuchos::rcp(
            new JouleHeatingSource<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "LorentzForce")
    {
        auto eval = Teuchos::rcp(
            new LorentzForce<EvalType, panzer::Traits, num_space_dim>(*ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    // Initialize 'error_msg' with list of closure models for Inductionless
    // MHD equations
    error_msg = "ElectricCurrentDensity\n";
    error_msg += "ElectricPotentialCrossProductFlux\n";
    error_msg += "ElectricPotentialDiffusionFlux\n";
    error_msg += "ExternalMagneticField\n";
    error_msg += "HartmannProblemExact\n";
    error_msg += "JouleHeatingSource\n";
    error_msg += "LorentzForce\n";
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_INDUCTIONLESSCLOSUREMODELFACTORY_IMPL_HPP
