#ifndef VERTEXCFD_EQUATIONSET_FULLINDUCTIONMHD_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_FULLINDUCTIONMHD_IMPL_HPP

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
FullInductionMHD<EvalType>::FullInductionMHD(
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

    // Check for fluid or solid domain to restrict input parameters accordingly
    const bool fluid_mhd = params->get<std::string>("Type")
                           == "FullInductionMHD";

    // Set default parameter values and validate the inputs.
    Teuchos::ParameterList valid_parameters;
    this->setDefaultValidParameters(valid_parameters);
    valid_parameters.set(
        "Model ID", "", "Closure model id associated with this equation set");
    valid_parameters.set("Basis Order", 1, "Order of the basis");
    valid_parameters.set("Integration Order", 2, "Order of the integration");
    valid_parameters.set(
        "Build Resistive Flux", false, "Resistive flux boolean");
    valid_parameters.set("Build Induction Constant Source",
                         false,
                         "Induction equation constant source boolean");
    valid_parameters.set("Build Magnetic Correction Potential Equation",
                         false,
                         "Solve full induction equation with divergence "
                         "cleaning");
    if (fluid_mhd)
    {
        valid_parameters.set("Build Divergence Cleaning Source",
                             false,
                             "Divergence cleaning source boolean");
        valid_parameters.set("Build Godunov-Powell Source",
                             false,
                             "Godunov-Powell source boolean");
    }
    valid_parameters.set("Build Magnetic Correction Damping Source",
                         false,
                         "Magnetic Correction damping source boolean");
    params->validateParametersAndSetDefaults(valid_parameters);

    // Extract parameters.
    const int basis_order = params->get<int>("Basis Order", 1);
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const std::string model_id = params->get<std::string>("Model ID");
    const bool build_ind_equ_const_source
        = params->get<bool>("Build Induction Constant Source");
    const bool build_magn_corr
        = params->get<bool>("Build Magnetic Correction Potential Equation");
    _build_resistive_flux = params->get<bool>("Build Resistive Flux");
    const bool build_div_cleaning_src
        = fluid_mhd ? params->get<bool>("Build Divergence Cleaning Source")
                    : false;
    const bool build_godunov_powell_source
        = fluid_mhd ? params->get<bool>("Build Godunov-Powell Source") : false;
    const bool build_magn_corr_damping_src
        = params->get<bool>("Build Magnetic Correction Damping Source");

    // convective flux is not evaluated for solid domains when there is no
    // divergence cleaning
    _build_convective_flux = fluid_mhd || build_magn_corr;

    // Initialize equation names and variables names for full induction model
    for (int d = 0; d < _num_space_dim; ++d)
    {
        const std::string ds = std::to_string(d);
        _equ_dof_fim_pair.insert(
            {"induction_" + ds, "induced_magnetic_field_" + ds});
    }

    if (build_magn_corr)
    {
        _equ_dof_fim_pair.insert(
            {"magnetic_correction_potential", "scalar_magnetic_potential"});
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

    // Setup degrees of freedom for FIM equations
    for (auto it : _equ_dof_fim_pair)
    {
        const auto equ_name = it.first;
        const auto dof_name = it.second;
        set_dofs(equ_name, dof_name);
        // Set up the source terms for induction equations
        if (std::string::npos != equ_name.find("induction"))
        {
            std::unordered_map<std::string, bool> source_term;
            source_term.insert(
                {"GODUNOV_POWELL_SOURCE", build_godunov_powell_source});
            source_term.insert({"CONSTANT_SOURCE", build_ind_equ_const_source});
            _equ_source_term.insert({equ_name, source_term});
        }
        if (std::string::npos != equ_name.find("magnetic_correction"))
        {
            std::unordered_map<std::string, bool> source_term;
            source_term.insert({"DIV_CLEANING_SOURCE", build_div_cleaning_src});
            source_term.insert({"DAMPING_SOURCE", build_magn_corr_damping_src});
            _equ_source_term.insert({equ_name, source_term});
        }
    }

    // Add closure models and set-up DOFs
    this->addClosureModel(model_id);
    this->setupDOFs();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void FullInductionMHD<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    // Integration data. The same rule and basis is used for all DOFs in this
    // implementation so use the lagrange pressure field to get this data.
    const auto it = _equ_dof_fim_pair.find("induction_0");
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

    // Build total residuals for FIM equations
    for (auto it : _equ_dof_fim_pair)
    {
        // Define local variables
        const auto eq_name = it.first;
        const auto dof_name = it.second;
        std::vector<std::string> residual_operator_names;

        // Add time residuals
        add_basis_time_scalar_residual(
            eq_name, "DQDT", 1.0, residual_operator_names);

        // Add convective fluxes
        if (_build_convective_flux)
        {
            add_grad_basis_time_residual(
                eq_name, "CONVECTIVE_FLUX", -1.0, residual_operator_names);
        }

        // Add resistive fluxes
        if (_build_resistive_flux)
        {
            add_grad_basis_time_residual(
                eq_name, "RESISTIVE_FLUX", 1.0, residual_operator_names);
        }

        // Add source terms
        for (const auto& [src_name, add_src] : _equ_source_term.at(eq_name))
        {
            if (add_src)
            {
                add_basis_time_scalar_residual(
                    eq_name, src_name, -1.0, residual_operator_names);
            }
        }

        // Build and register residuals
        this->buildAndRegisterResidualSummationEvaluator(
            fm, dof_name, residual_operator_names, "RESIDUAL_" + eq_name);
    }
}

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_FULLINDUCTIONMHD_IMPL_HPP
