#ifndef VERTEXCFD_EQUATIONSET_SOLIDINDUCTIONLESSMHD_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_SOLIDINDUCTIONLESSMHD_IMPL_HPP

#include <Panzer_BasisIRLayout.hpp>
#include <Panzer_EvaluatorStyle.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Integrator_GradBasisDotVector.hpp>

#include <stdexcept>
#include <vector>

namespace VertexCFD
{
namespace EquationSet
{
//---------------------------------------------------------------------------//
template<class EvalType>
SolidInductionLessMHD<EvalType>::SolidInductionLessMHD(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const int& default_integration_order,
    const panzer::CellData& cell_data,
    const Teuchos::RCP<panzer::GlobalData>& global_data,
    const bool build_transient_support)
    : panzer::EquationSet_DefaultImpl<EvalType>(params,
                                                default_integration_order,
                                                cell_data,
                                                global_data,
                                                build_transient_support)
    , _dof_name("electric_potential")
    , _equ_name("electric_potential_equation")
{
    // Set default parameter values and validate the inputs.
    Teuchos::ParameterList valid_parameters;
    this->setDefaultValidParameters(valid_parameters);
    valid_parameters.set(
        "Model ID", "", "Closure model id associated with this equation set");
    valid_parameters.set("Basis Order", 1, "Order of the basis");
    valid_parameters.set("Integration Order", 2, "Order of the integration");
    valid_parameters.set(
        "Build Source Residual", false, "Build source residual");
    params->validateParametersAndSetDefaults(valid_parameters);

    // Extract parameters.
    const int basis_order = params->get<int>("Basis Order");
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const auto model_id = params->get<std::string>("Model ID");
    _build_source_term = params->get<bool>("Build Source Residual");

    // Set up degrees of freedom for solid induction-less MHD equation
    const std::string residual_name = "RESIDUAL_" + _equ_name;
    const std::string scatter_name = "SCATTER_" + _equ_name;
    const std::string basis_type = "HGrad";
    this->addDOF(_dof_name,
                 basis_type,
                 basis_order,
                 integration_order,
                 residual_name,
                 scatter_name);
    this->addDOFGrad(_dof_name);

    // Add closure models and set-up DOFs
    this->addClosureModel(model_id);
    this->setupDOFs();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void SolidInductionLessMHD<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    // Get integration rule and basis function for field
    const auto ir = this->getIntRuleForDOF(_dof_name);
    const auto basis = this->getBasisIRLayoutForDOF(_dof_name);

    // Add grad-basis time residuals
    auto add_grad_basis_time_residual = [&, this](
                                            const std::string& equ_name,
                                            const std::string& residual_name,
                                            const double& multiplier,
                                            std::vector<std::string>& op_names) {
        const std::string closure_model_residual = residual_name + "_"
                                                   + equ_name;
        const std::string full_residual = "RESIDUAL_" + closure_model_residual;
        auto op = Teuchos::rcp(
            new panzer::Integrator_GradBasisDotVector<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                full_residual,
                closure_model_residual,
                *basis,
                *ir,
                multiplier));
        this->template registerEvaluator<EvalType>(fm, op);
        op_names.push_back(full_residual);
    };

    // Collect each term for residual to form solid induction-less MHD equation
    std::vector<std::string> residual_operator_names;

    // Second-order flux: heat flux
    add_grad_basis_time_residual(
        _equ_name, "ELECTRIC_POTENTIAL_FLUX", 1.0, residual_operator_names);

    // Build total residual for solid induction-less MHD equation
    this->buildAndRegisterResidualSummationEvaluator(fm,
                                                     _dof_name,
                                                     residual_operator_names,
                                                     "RESIDUAL_electric_"
                                                     "potential_equation");
}

//---------------------------------------------------------------------------/
template<class EvalType>
std::string SolidInductionLessMHD<EvalType>::fieldName(const int dof) const
{
    if (dof > 0)
    {
        throw std::logic_error(
            "SolidInductionLessMHD equation contributes a single DOF");
    }

    return _dof_name;
}

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_SOLIDINDUCTIONLESSMHD_IMPL_HPP
