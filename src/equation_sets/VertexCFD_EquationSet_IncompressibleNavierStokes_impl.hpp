#ifndef VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_NAVIERSTOKES_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_NAVIERSTOKES_IMPL_HPP

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
IncompressibleNavierStokes<EvalType>::IncompressibleNavierStokes(
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

    // Set default parameter values and validate the inputs.
    Teuchos::ParameterList valid_parameters;
    this->setDefaultValidParameters(valid_parameters);
    valid_parameters.set(
        "Model ID", "", "Closure model id associated with this equation set");
    valid_parameters.set("Basis Order", 1, "Order of the basis");
    valid_parameters.set("Integration Order", 2, "Order of the integration");
    valid_parameters.set("Build Viscous Flux", true, "Viscous flux boolean");
    valid_parameters.set(
        "Build Temperature Equation", false, "Solve temperature equation");
    valid_parameters.set("Build Inductionless MHD Equation",
                         false,
                         "Solve electric potential equation");
    valid_parameters.set(
        "Build Full Induction Model", false, "Solve full induction equation");
    valid_parameters.set("Build Magnetic Correction Potential Equation",
                         false,
                         "Solve full induction equation with divergence "
                         "cleaning");
    valid_parameters.set(
        "Build Constant Source", false, "Constant source boolean");

    valid_parameters.set(
        "Build Buoyancy Source", false, "Buoyancy source boolean");

    valid_parameters.set("Build Viscous Heat", false, "Viscous heat boolean");

    valid_parameters.set(
        "Build Resistive Flux", false, "Resistive flux boolean");

    valid_parameters.set("Build Divergence Cleaning Source",
                         false,
                         "Divergence cleaning source boolean");

    valid_parameters.set(
        "Build Godunov-Powell Source", false, "Godunov-Powell source boolean");

    valid_parameters.set("Build Magnetic Correction Damping Source",
                         false,
                         "Magnetic Correction damping source boolean");

    valid_parameters.set("Build Induction Constant Source",
                         false,
                         "Induction equation constant source boolean");

    const auto turbulence_validator = Teuchos::rcp(new Teuchos::StringValidator(
        Teuchos::tuple<std::string>("No Turbulence Model",
                                    "Spalart-Allmaras",
                                    "Standard K-Epsilon",
                                    "Realizable K-Epsilon",
                                    "K-Omega",
                                    "WALE")));

    valid_parameters.set("Turbulence Model",
                         "No Turbulence Model",
                         "Turbulence model choice",
                         turbulence_validator);

    params->validateParametersAndSetDefaults(valid_parameters);

    // Extract parameters.
    const int basis_order = params->get<int>("Basis Order", 1);
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const std::string model_id = params->get<std::string>("Model ID");
    _build_viscous_flux = params->get<bool>("Build Viscous Flux", true);
    _build_temp_equ = params->get<bool>("Build Temperature Equation", false);
    _build_ind_less_equ
        = params->get<bool>("Build Inductionless MHD Equation", false);
    _build_constant_source = params->get<bool>("Build Constant Source", false);
    _build_buoyancy_source = params->get<bool>("Build Buoyancy Source", false);
    _build_viscous_heat = params->get<bool>("Build Viscous Heat", false);
    _turbulence_model = params->get<std::string>("Turbulence Model");
    _build_full_induction_model
        = params->get<bool>("Build Full Induction Model", false);
    _build_magn_corr = params->get<bool>(
        "Build Magnetic Correction Potential Equation", false);
    _build_resistive_flux = params->get<bool>("Build Resistive Flux", false);
    const bool build_div_cleaning_src
        = params->get<bool>("Build Divergence Cleaning Source", false);
    _build_godunov_powell_source
        = params->get<bool>("Build Godunov-Powell Source", false);
    const bool build_magn_corr_damping_src
        = params->get<bool>("Build Magnetic Correction Damping Source");
    const bool build_ind_equ_const_source
        = params->get<bool>("Build Induction Constant Source");

    if (_build_buoyancy_source && !_build_temp_equ)
    {
        std::string msg = "Build Buoyancy Source set to TRUE,\n";
        msg += "but Build Temperature Equation set to FALSE.\n";
        msg += "Please enable temperature equation to solve with buoyancy source.";
        throw std::runtime_error(msg);
    }

    if (_build_viscous_heat && !_build_temp_equ)
    {
        std::string msg = "Build Viscous Heat set to TRUE.\n";
        msg += "but Build Temperature Equation set to FALSE.\n";
        msg += "Please enable temperature equation to solve with buoyancy source.";
        throw std::runtime_error(msg);
    }

    // Initialize equation names and variable names for NS equations
    _equ_dof_ns_pair.insert({"continuity", "lagrange_pressure"});
    for (int d = 0; d < _num_space_dim; ++d)
    {
        const std::string ds = std::to_string(d);
        _equ_dof_ns_pair.insert({"momentum_" + ds, "velocity_" + ds});
    }
    if (_build_temp_equ)
        _equ_dof_ns_pair.insert({"energy", "temperature"});

    // Initialize equation name and variable name for EP equation
    if (_build_ind_less_equ)
        _equ_dof_ep_pair.insert(
            {"electric_potential_equation", "electric_potential"});

    // Initialize equation names and variable names for turbulence model (TM)
    // equations
    if (std::string::npos != _turbulence_model.find("Spalart-Allmaras"))
    {
        _equ_dof_tm_pair.insert(
            {"spalart_allmaras_equation", "spalart_allmaras_variable"});
    }
    else if (std::string::npos != _turbulence_model.find("K-Epsilon"))
    {
        _equ_dof_tm_pair.insert(
            {"turb_kinetic_energy_equation", "turb_kinetic_energy"});
        _equ_dof_tm_pair.insert(
            {"turb_dissipation_rate_equation", "turb_dissipation_rate"});
    }
    else if (std::string::npos != _turbulence_model.find("K-Omega"))
    {
        _equ_dof_tm_pair.insert(
            {"turb_kinetic_energy_equation", "turb_kinetic_energy"});
        _equ_dof_tm_pair.insert({"turb_specific_dissipation_rate_equation",
                                 "turb_specific_dissipation_rate"});
    }

    // Initialize equation names and variables names for full induction model
    if (_build_full_induction_model)
    {
        for (int d = 0; d < _num_space_dim; ++d)
        {
            const std::string ds = std::to_string(d);
            _equ_dof_fim_pair.insert(
                {"induction_" + ds, "induced_magnetic_field_" + ds});
        }

        if (_build_magn_corr)
        {
            _equ_dof_fim_pair.insert({"magnetic_correction_potential",
                                      "scalar_magnetic_potential"});
        }
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

    // Setup degrees of freedom for NS equations.
    for (auto it : _equ_dof_ns_pair)
    {
        const auto equ_name = it.first;
        const auto dof_name = it.second;
        set_dofs(equ_name, dof_name);
    };

    // Setup degrees of freedom for EP equations.
    for (auto it : _equ_dof_ep_pair)
    {
        const auto equ_name = it.first;
        const auto dof_name = it.second;
        set_dofs(equ_name, dof_name);
    }

    // Setup degrees of freedom for TM equations.
    for (auto it : _equ_dof_tm_pair)
    {
        const auto equ_name = it.first;
        const auto dof_name = it.second;
        set_dofs(equ_name, dof_name);
    }

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
                {"GODUNOV_POWELL_SOURCE", _build_godunov_powell_source});
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
void IncompressibleNavierStokes<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    // Integration data. The same rule and basis is used for all DOFs in this
    // implementation so use the lagrange pressure field to get this data.
    const auto it = _equ_dof_ns_pair.find("continuity");
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

    // Build total residuals for NS equations
    for (auto it : _equ_dof_ns_pair)
    {
        // Define local variables
        const auto equ_name = it.first;
        const auto dof_name = it.second;
        std::vector<std::string> residual_operator_names;

        // Time derivative residual
        add_basis_time_scalar_residual(
            equ_name, "DQDT", 1.0, residual_operator_names);

        // Convective flux residual
        add_grad_basis_time_residual(
            equ_name, "CONVECTIVE_FLUX", -1.0, residual_operator_names);

        // Viscous flux residual
        if (_build_viscous_flux)
        {
            add_grad_basis_time_residual(
                equ_name, "VISCOUS_FLUX", 1.0, residual_operator_names);
        }

        // Constant source residual
        if (_build_constant_source)
        {
            add_basis_time_scalar_residual(
                equ_name, "CONSTANT_SOURCE", -1.0, residual_operator_names);
        }

        // Buoyancy source residual
        if (_build_buoyancy_source)
        {
            add_basis_time_scalar_residual(
                equ_name, "BUOYANCY_SOURCE", -1.0, residual_operator_names);
        }

        // Viscous heating source residual
        if (_build_viscous_heat)
        {
            add_basis_time_scalar_residual(
                equ_name, "VISCOUS_HEAT", -1.0, residual_operator_names);
        }

        // Lorentz force for momentum equations
        if (_build_ind_less_equ
            && std::string::npos != equ_name.find("momentum"))
        {
            add_basis_time_scalar_residual(
                equ_name, "VOLUMETRIC_SOURCE", -1.0, residual_operator_names);
        }

        // Godunov-Powell source for momentum equations in full induction MHD
        if (_build_full_induction_model && _build_godunov_powell_source
            && std::string::npos != equ_name.find("momentum"))
        {
            add_basis_time_scalar_residual(equ_name,
                                           "GODUNOV_POWELL_SOURCE",
                                           -1.0,
                                           residual_operator_names);
        }

        // Build and register residuals
        this->buildAndRegisterResidualSummationEvaluator(
            fm, dof_name, residual_operator_names, "RESIDUAL_" + equ_name);
    }

    // Build total residual for EP equation
    for (auto it : _equ_dof_ep_pair)
    {
        // Define local variables
        const auto equ_name = it.first;
        const auto dof_name = it.second;
        std::vector<std::string> residual_operator_names;

        // Cross-product flux and diffusion flux residuals
        add_grad_basis_time_residual(
            equ_name, "ELECTRIC_POTENTIAL_FLUX", 1.0, residual_operator_names);

        // Build and register residuals
        this->buildAndRegisterResidualSummationEvaluator(
            fm, dof_name, residual_operator_names, "RESIDUAL_" + equ_name);
    }

    // Build total residuals for TM equations
    for (auto it : _equ_dof_tm_pair)
    {
        // Define local variables
        const auto eq_name = it.first;
        const auto dof_name = it.second;
        std::vector<std::string> residual_operator_names;

        // Add time, convective, viscous and source residuals
        add_basis_time_scalar_residual(
            eq_name, "DQDT", 1.0, residual_operator_names);
        add_grad_basis_time_residual(
            eq_name, "CONVECTIVE_FLUX", -1.0, residual_operator_names);
        add_grad_basis_time_residual(
            eq_name, "DIFFUSION_FLUX", 1.0, residual_operator_names);
        add_basis_time_scalar_residual(
            eq_name, "SOURCE", -1.0, residual_operator_names);

        // Build and register residuals
        this->buildAndRegisterResidualSummationEvaluator(
            fm, dof_name, residual_operator_names, "RESIDUAL_" + eq_name);
    }

    // Build total residuals for FIM equations
    for (auto it : _equ_dof_fim_pair)
    {
        // Define local variables
        const auto eq_name = it.first;
        const auto dof_name = it.second;
        std::vector<std::string> residual_operator_names;

        // Add time and convective residuals
        add_basis_time_scalar_residual(
            eq_name, "DQDT", 1.0, residual_operator_names);
        add_grad_basis_time_residual(
            eq_name, "CONVECTIVE_FLUX", -1.0, residual_operator_names);

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

#endif // end VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_NAVIERSTOKES_IMPL_HPP
