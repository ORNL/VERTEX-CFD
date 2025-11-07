#ifndef VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_LSVOF_IMPL_HPP
#define VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_LSVOF_IMPL_HPP

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
IncompressibleLSVOF<EvalType>::IncompressibleLSVOF(
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

    // Extract basic model parameters
    const int basis_order = params->get<int>("Basis Order", 1);
    const int integration_order
        = params->get<int>("Integration Order", basis_order + 1);
    const std::string model_id = params->get<std::string>("Model ID");

    // LSVOF parameters
    Teuchos::ParameterList& lsvof_params = params->sublist("LSVOF_Properties");

    _build_lsvofmom_equ
        = params->get<bool>("Build LSVOF Navier-Stokes Equations", true);

    _lsvof_model_name = lsvof_params.get<std::string>("LSVOF Model");

    const int num_phases = lsvof_params.get<int>("Number of Phases");

    _build_lsvof_buoyancy_source
        = lsvof_params.get<bool>("Build LSVOF Buoyancy Source", true);

    _build_lsvof_surface_tension
        = lsvof_params.get<bool>("Build LSVOF Surface Tension", true);

    // Initialize equation names and variable names for NS equations
    if (_build_lsvofmom_equ)
    {
        _equ_dof_ns_pair.insert({"continuity", "lagrange_pressure"});

        for (int d = 0; d < _num_space_dim; ++d)
        {
            const std::string ds = std::to_string(d);
            _equ_dof_ns_pair.insert({"momentum_" + ds, "velocity_" + ds});
        }
    }

    // Initialize equation name and variable names for LSVOF equations
    if (_lsvof_model_name == "VOF")
    {
        for (int n = 1; n < num_phases; ++n)
        {
            const std::string list_name = "Phase " + std::to_string(n);

            Teuchos::ParameterList phase_list = lsvof_params.sublist(list_name);

            const std::string phase_name
                = phase_list.get<std::string>("Phase Name");

            const std::string phase_fraction_name = "alpha_" + phase_name;

            _equ_dof_lsvof_pair.insert(
                {phase_fraction_name + "_equation", phase_fraction_name});
        }
    }
    else if (_lsvof_model_name == "CLS")
    {
        _equ_dof_lsvof_pair.insert({"level_set_equation", "level_set"});
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
    if (_build_lsvofmom_equ)
    {
        for (const auto& [equ_name, dof_name] : _equ_dof_ns_pair)
        {
            set_dofs(equ_name, dof_name);
        }
    }

    // Setup degrees of freedom for LSVOF equations.
    for (const auto& [equ_name, dof_name] : _equ_dof_lsvof_pair)
    {
        set_dofs(equ_name, dof_name);
    }

    // Add closure models and set-up DOFs
    this->addClosureModel(model_id);
    this->setupDOFs();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void IncompressibleLSVOF<EvalType>::buildAndRegisterEquationSetEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::FieldLibrary&,
    const Teuchos::ParameterList&) const
{
    // Integration data. The same rule and basis is used for all DOFs in this
    // implementation so use the first entry of `_equ_dof_lsvof_pair` to get
    // this data.
    const auto it = _equ_dof_lsvof_pair.begin();
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
    if (_build_lsvofmom_equ)
    {
        for (const auto& [equ_name, dof_name] : _equ_dof_ns_pair)
        {
            std::vector<std::string> residual_operator_names;

            // Time derivative residual
            add_basis_time_scalar_residual(
                equ_name, "DQDT", 1.0, residual_operator_names);

            // Convective flux residual
            add_grad_basis_time_residual(
                equ_name, "CONVECTIVE_FLUX", -1.0, residual_operator_names);

            // Viscous flux residual
            add_grad_basis_time_residual(
                equ_name, "VISCOUS_FLUX", 1.0, residual_operator_names);

            // TODO: add body force, surface tension, etc. to the total
            // residual as they are created

            if (std::string::npos != equ_name.find("momentum"))
            {
                // Buoyancy source residual for LSVOF
                if (_build_lsvof_buoyancy_source)
                {
                    add_basis_time_scalar_residual(equ_name,
                                                   "LSVOF_BUOYANCY_SOURCE",
                                                   -1.0,
                                                   residual_operator_names);
                }

                if (_build_lsvof_surface_tension)
                {
                    add_grad_basis_time_residual(equ_name,
                                                 "SURFACE_TENSION_FLUX",
                                                 1.0,
                                                 residual_operator_names);
                }
            }

            // Build and register residuals
            this->buildAndRegisterResidualSummationEvaluator(
                fm, dof_name, residual_operator_names, "RESIDUAL_" + equ_name);
        }
    }

    // Build total residuals for LSVOF equations
    for (const auto& [equ_name, dof_name] : _equ_dof_lsvof_pair)
    {
        std::vector<std::string> residual_operator_names;

        // Add time and convective flux residuals
        add_basis_time_scalar_residual(
            equ_name, "DQDT", 1.0, residual_operator_names);

        add_grad_basis_time_residual(
            equ_name, "CONVECTIVE_FLUX", -1.0, residual_operator_names);

        if (_lsvof_model_name == "VOF")
        {
            add_grad_basis_time_residual(equ_name,
                                         "ARTIFICIAL_COMPRESSION",
                                         -1.0,
                                         residual_operator_names);
        }
        else if (_lsvof_model_name == "CLS")
        {
            add_grad_basis_time_residual(
                equ_name, "REG", -1.0, residual_operator_names);
        }

        // TODO: add compressive flux, source terms, etc. to total residual
        // as they are created

        // Build and register residuals
        this->buildAndRegisterResidualSummationEvaluator(
            fm, dof_name, residual_operator_names, "RESIDUAL_" + equ_name);
    }
}

//---------------------------------------------------------------------------//

} // end namespace EquationSet
} // end namespace VertexCFD

#endif // end VERTEXCFD_EQUATIONSET_INCOMPRESSIBLE_LSVOF_IMPL_HPP
