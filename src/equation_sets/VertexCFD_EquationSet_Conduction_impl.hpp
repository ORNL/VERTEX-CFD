#ifndef VERTEXCFD_EQUATIONSET_CONDUCTION_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_CONDUCTION_IMPL_HPP

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
Conduction<EvalType>::Conduction(
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
    , _dof_name("temperature")
    , _equ_name("energy")
{
    // This equation set need not always be transient. Could solve Poisson
    if (!this->buildTransientSupport())
    {
        throw std::logic_error(
            "Conduction equation requires transient support");
    }

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
    const int basis_order = params->get<int>("Basis Order", 1);
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const auto model_id = params->get<std::string>("Model ID");
    _build_source_term = params->get<bool>("Build Source Residual");

    // Set up degrees of freedom for conduction equation
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
    if (this->buildTransientSupport())
    {
        this->addDOFTimeDerivative(_dof_name);
    }

    // Add closure models and set-up DOFs
    this->addClosureModel(model_id);
    this->setupDOFs();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void Conduction<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    // Get integration rule and basis function for field
    const auto ir = this->getIntRuleForDOF(_dof_name);
    const auto basis = this->getBasisIRLayoutForDOF(_dof_name);

    // Add basis time residuals
    auto add_basis_time_scalar_residual = [&, this](
                                              const std::string& equ_name,
                                              const std::string& residual_name,
                                              const double& multiplier,
                                              std::vector<std::string>& op_names) {
        const std::string closure_model_residual = residual_name + "_"
                                                   + equ_name;
        const std::string full_residual = "RESIDUAL_" + closure_model_residual;
        auto op = Teuchos::rcp(
            new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                full_residual,
                closure_model_residual,
                *basis,
                *ir,
                multiplier));
        this->template registerEvaluator<EvalType>(fm, op);
        op_names.push_back(full_residual);
    };

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

    // Collect each term for residual to form conduction equation
    std::vector<std::string> residual_operator_names;

    // Time derivative residual
    add_basis_time_scalar_residual(
        _equ_name, "DQDT", 1.0, residual_operator_names);

    // Second-order flux: heat flux
    add_grad_basis_time_residual(
        _equ_name, "CONDUCTION_FLUX", 1.0, residual_operator_names);

    // Source term
    if (_build_source_term)
    {
        add_basis_time_scalar_residual(
            _equ_name, "SOURCE", -1.0, residual_operator_names);
    }

    // Build total residual for conduction equation
    this->buildAndRegisterResidualSummationEvaluator(
        fm, _dof_name, residual_operator_names, "RESIDUAL_energy");
}

//---------------------------------------------------------------------------/
template<class EvalType>
std::string Conduction<EvalType>::fieldName(const int dof) const
{
    if (dof > 0)
    {
        throw std::logic_error("Conduction equation contributes a single DOF");
    }

    return _dof_name;
}

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_CONDUCTION_IMPL_HPP
