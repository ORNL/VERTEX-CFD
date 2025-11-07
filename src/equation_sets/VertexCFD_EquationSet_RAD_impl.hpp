#ifndef VERTEXCFD_EQUATIONSET_RAD_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_RAD_IMPL_HPP

#include <Panzer_BasisIRLayout.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Integrator_GradBasisDotVector.hpp>

#include <Phalanx_FieldManager.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace EquationSet
{
//---------------------------------------------------------------------------//
template<class EvalType>
RAD<EvalType>::RAD(const Teuchos::RCP<Teuchos::ParameterList>& params,
                   const int& default_integration_order,
                   const panzer::CellData& cell_data,
                   const Teuchos::RCP<panzer::GlobalData>& global_data,
                   const bool build_transient_support)
    : panzer::EquationSet_DefaultImpl<EvalType>(params,
                                                default_integration_order,
                                                cell_data,
                                                global_data,
                                                build_transient_support)
{
    // This equation set is always transient.
    if (!this->buildTransientSupport())
    {
        throw std::logic_error(
            "Transient support is required for solving all equation sets.");
    }

    // Get the number of space dimensions.
    _num_space_dim = cell_data.baseCellDimension();
    if (!(_num_space_dim == 2 || _num_space_dim == 3))
    {
        throw std::runtime_error("Number of space dimensions not supported");
    }

    // Set default parameter values and validate the inputs.
    Teuchos::ParameterList valid_parameters;
    this->setDefaultValidParameters(valid_parameters);
    valid_parameters.set(
        "Model ID", "", "Closure model id associated with this equation set");
    valid_parameters.set("Basis Order", 1, "Order of the basis");
    valid_parameters.set("Integration Order", 2, "Order of the integration");
    valid_parameters.set("Build Reaction", false, "Reaction boolean");
    valid_parameters.set("Build Advection", false, "Advection boolean");
    valid_parameters.set("Build Diffusion", false, "Diffusion boolean");
    valid_parameters.set(
        "Build Fission Source", false, "Fission source boolean");
    valid_parameters.set("Number of Species", 1, "Number of species");

    params->validateParametersAndSetDefaults(valid_parameters);

    // Extract parameters.
    const int basis_order = params->get<int>("Basis Order", 1);
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const std::string model_id = params->get<std::string>("Model ID");
    _build_reaction = params->get<bool>("Build Reaction", false);
    _build_advection = params->get<bool>("Build Advection", false);
    _build_diffusion = params->get<bool>("Build Diffusion", false);
    _build_fission_source = params->get<bool>("Build Fission Source", false);
    const int _num_species = params->get<int>("Number of Species", 1);

    // Initialize equation names and variable names for
    // reaction-advection-diffusion equations
    for (int d = 0; d < _num_species; ++d)
    {
        const std::string ds = std::to_string(d);
        _equ_dof_rad_pair.insert(
            {"reaction_advection_diffusion_equation_" + ds, "species_" + ds});
    }

    // Functions to set dofs
    auto set_dofs
        = [&, this](const std::string& equ_name, const std::string& dof_name) {
              const std::string residual_name = "RESIDUAL_" + equ_name;
              const std::string scatter_name = "SCATTER_" + equ_name;
              const std::string basis_type = "HGrad";
              this->addDOF(dof_name,
                           basis_type,
                           basis_order,
                           integration_order,
                           residual_name,
                           scatter_name);
              this->addDOFGrad(dof_name);
              this->addDOFTimeDerivative(dof_name);
          };

    // Setup degrees of freedom for reaction-advection equations.
    for (const auto& [equ_name, dof_name] : _equ_dof_rad_pair)
    {
        set_dofs(equ_name, dof_name);
    };

    // Add closure models and set-up DOFs
    this->addClosureModel(model_id);
    this->setupDOFs();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void RAD<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    // Integration data. The same rule and basis is used for all DOFs in this
    // implementation so use the first one to get this data.
    const auto it = _equ_dof_rad_pair.begin();
    const auto ir = this->getIntRuleForDOF(it->second);
    const auto basis = this->getBasisIRLayoutForDOF(it->second);

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

    // Build total residuals for reaction-advection equations
    for (const auto& [equ_name, dof_name] : _equ_dof_rad_pair)
    {
        // Define local variables
        std::vector<std::string> residual_operator_names;

        // Time derivative residual
        add_basis_time_scalar_residual(
            equ_name, "DQDT", 1.0, residual_operator_names);

        // Reaction term residual
        if (_build_reaction)
        {
            add_basis_time_scalar_residual(
                equ_name, "REACTION_TERM", -1.0, residual_operator_names);
        }

        // Advection flux residual
        if (_build_advection)
        {
            add_grad_basis_time_residual(
                equ_name, "ADVECTION_FLUX", -1.0, residual_operator_names);
        }

        // Diffusion flux residual
        if (_build_diffusion)
        {
            add_grad_basis_time_residual(
                equ_name, "DIFFUSION_FLUX", 1.0, residual_operator_names);
        }

        // Fission source residual
        if (_build_fission_source)
        {
            add_basis_time_scalar_residual(
                equ_name, "FISSION_SOURCE", -1.0, residual_operator_names);
        }

        // Build and register residuals
        this->buildAndRegisterResidualSummationEvaluator(
            fm, dof_name, residual_operator_names, "RESIDUAL_" + equ_name);
    }
}

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_RAD_IMPL_HPP
