#ifndef VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_IMPL_HPP

#include "VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include "closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"
#include "closure_models/VertexCFD_Closure_VariableConvectiveFlux.hpp"
#include "closure_models/VertexCFD_Closure_VariableDiffusionFlux.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleWallFunctionStress.hpp"
#include "incompressible_solver/boundary_conditions/VertexCFD_IncompressibleBoundaryState_Factory.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleGradDiv.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousFlux.hpp"

#include "incompressible_solver/fluid_properties/VertexCFD_Closure_IncompressibleFluidProperties.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_ElectricPotentialBoundaryState_Factory.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialCrossProductFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialDiffusionFlux.hpp"

#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceBoundaryEddyViscosity.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_TurbulenceBoundaryState_Factory.hpp"

#include <Panzer_DOF.hpp>
#include <Panzer_DOFGradient.hpp>
#include <Panzer_DotProduct.hpp>
#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Normals.hpp>
#include <Panzer_Sum.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_MDField.hpp>

#include <map>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::IncompressibleBoundaryFlux(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : BoundaryFluxBase<EvalType, NumSpaceDim>(bc, global_data)
{
    // Check if boundary is an internal interface between solid and fluid
    // regions
    _internal_interface = bc.params()->isType<bool>("Fluid/Solid Interface")
                              ? bc.params()->get<bool>("Fluid/Solid Interface")
                              : false;
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::setup(
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& user_data)
{
    // Temperature equation boolean
    const std::string continuity_model
        = user_data.isType<std::string>("Continuity Model")
              ? user_data.get<std::string>("Continuity Model")
              : "AC";

    if (continuity_model == "AC")
    {
        _continuity_model = ConModel::AC;
    }
    else if (continuity_model == "EDAC")
    {
        _continuity_model = ConModel::EDAC;
    }
    else if (continuity_model == "EDACTempNC")
    {
        _continuity_model = ConModel::EDACTempNC;
    }

    // Viscous flux boolean.
    _build_viscous_flux = user_data.get<bool>("Build Viscous Flux");

    // Temperature equation boolean
    _build_temp_equ = user_data.isType<bool>("Build Temperature Equation")
                          ? user_data.get<bool>("Build Temperature Equation")
                          : false;

    // Electric potential equation boolean
    _build_ind_less_equ
        = user_data.isType<bool>("Build Inductionless MHD Equation")
              ? user_data.get<bool>("Build Inductionless MHD Equation")
              : false;

    // Internal interface case: temperature and electric potential equations do
    // not need boundary conditions.
    if (_internal_interface)
    {
        _build_temp_equ = false;
        _build_ind_less_equ = false;
    }

    // Turbulence model boolean
    _turbulence_model_name
        = user_data.isType<std::string>("Turbulence Model")
              ? user_data.get<std::string>("Turbulence Model")
              : "No Turbulence Model";
    _turbulence_model = _turbulence_model_name == "No Turbulence Model" ? false
                                                                        : true;

    // Initialize equation names and variable names for NS equations
    // and temperature equation. Note that temperature equation is not
    // added to the list if there is a solid/fluid interface.
    _equ_dof_ns_pair.insert({"continuity", "lagrange_pressure"});
    for (int d = 0; d < num_space_dim; ++d)
    {
        const std::string ds = std::to_string(d);
        _equ_dof_ns_pair.insert({"momentum_" + ds, "velocity_" + ds});
    }
    if (_build_temp_equ)
        _equ_dof_ns_pair.insert({"energy", "temperature"});

    // Initialize equation name and variable name for EP equation
    if (_build_ind_less_equ)
    {
        _equ_dof_ep_pair.insert(
            {"electric_potential_equation", "electric_potential"});
    }

    // Initialize equation name and variable name for TM equations
    if (_turbulence_model)
    {
        if (std::string::npos
            != _turbulence_model_name.find("Spalart-Allmaras"))
        {
            _equ_dof_tm_pair.insert(
                {"spalart_allmaras_equation", "spalart_allmaras_variable"});
        }
        else if (std::string::npos != _turbulence_model_name.find("K-Epsilon"))
        {
            _equ_dof_tm_pair.insert(
                {"turb_kinetic_energy_equation", "turb_kinetic_energy"});
            _equ_dof_tm_pair.insert(
                {"turb_dissipation_rate_equation", "turb_dissipation_rate"});
        }
        else if (std::string::npos != _turbulence_model_name.find("K-Omega"))
        {
            _equ_dof_tm_pair.insert(
                {"turb_kinetic_energy_equation", "turb_kinetic_energy"});
            _equ_dof_tm_pair.insert({"turb_specific_dissipation_rate_equation",
                                     "turb_specific_dissipation_rate"});
        }
        else if (std::string::npos != _turbulence_model_name.find("SST"))
        {
            _equ_dof_tm_pair.insert(
                {"turb_kinetic_energy_equation", "turb_kinetic_energy"});
            _equ_dof_tm_pair.insert({"turb_specific_dissipation_rate_equation",
                                     "turb_specific_dissipation_rate"});
        }
        else if (std::string::npos != _turbulence_model_name.find("K-Tau"))
        {
            _equ_dof_tm_pair.insert(
                {"turb_kinetic_energy_equation", "turb_kinetic_energy"});
            _equ_dof_tm_pair.insert({"turb_specific_dissipation_rate_equation",
                                     "turb_specific_dissipation_rate"});
        }
    }

    // Initialize parent class variables (only needed with one set of
    // equations)
    this->initialize(side_pb, _equ_dof_ns_pair);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::buildAndRegisterEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&,
    const Teuchos::ParameterList& closure_models,
    const Teuchos::ParameterList& user_data) const
{
    // Map to store residuals for each equation listed in `_equ_dof_ns_pair`
    std::unordered_map<std::string, std::vector<std::string>> eq_vct_map;

    // Get integration rule for closure models
    const auto ir = this->integrationRule();

    // Create degree of freedom and gradients for NS equations
    for (auto& pair : _equ_dof_ns_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Create degree of freedom and gradients for EP equations
    for (auto& pair : _equ_dof_ep_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Create degree of freedom and gradients for TM equations
    for (auto& pair : _equ_dof_tm_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Register normals
    this->registerSideNormals(fm, side_pb);

    // Fluid properties: model id is stored in a sublist called `child0`.
    const std::string model_id
        = side_pb.getParameterList()->sublist("child0").get<std::string>(
            "Model ID");

    // Stabilization method: model id is stored in a sublist called `child0`.
    const std::string stabilization_method
        = side_pb.getParameterList()->sublist("child0").get<std::string>(
            "Stabilization Method");

    // Define the stability parameter list as an empty list then populate the
    // list if parameter list exists in the input file.
    Teuchos::ParameterList stability_param_list;
    if (closure_models.sublist(model_id).isSublist("Stability Parameters"))
    {
        stability_param_list
            = closure_models.sublist(model_id).sublist("Stability Parameters");
    }

    // Fluid properties
    Teuchos::ParameterList fluid_prop_list
        = closure_models.sublist(model_id).sublist("Fluid Properties");

    const bool build_buoyancy
        = user_data.isType<bool>("Build Buoyancy Source")
              ? user_data.get<bool>("Build Buoyancy Source")
              : false;

    fluid_prop_list.set<bool>("Build Temperature Equation", _build_temp_equ);
    fluid_prop_list.set<bool>("Build Buoyancy Source", build_buoyancy);
    fluid_prop_list.set<bool>("Build Inductionless MHD Equation",
                              _build_ind_less_equ);

    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    auto eval = Teuchos::rcp(
        new FluidProperties::IncompressibleFluidProperties<EvalType,
                                                           panzer::Traits>(
            *ir, fluid_prop_list));
    this->template registerEvaluator<EvalType>(fm, eval);

    // Create boundary state operators for NS equations and EP equation
    // Get bc sublist
    const auto bc_params = *(this->m_bc.params());

    // NS equations
    const auto ns_bc_sublist = bc_params.isSublist("Navier-Stokes")
                                   ? bc_params.sublist("Navier-Stokes")
                                   : bc_params;
    auto incomp_ns_boundary_state_op
        = IncompressibleBoundaryStateFactory<EvalType,
                                             panzer::Traits,
                                             num_space_dim>::create(*ir,
                                                                    ns_bc_sublist,
                                                                    user_data,
                                                                    fluid_prop);
    this->template registerEvaluator<EvalType>(fm, incomp_ns_boundary_state_op);

    // EP equations
    if (_build_ind_less_equ)
    {
        const auto ep_bc_sublist = bc_params.sublist("Electric Potential");
        auto ep_boundary_state_op = ElectricPotentialBoundaryStateFactory<
            EvalType,
            panzer::Traits,
            num_space_dim>::create(*ir, ep_bc_sublist, user_data);
        this->template registerEvaluator<EvalType>(fm, ep_boundary_state_op);
    }

    // First-order flux //

    // Create boundary convective fluxes for NS equations
    auto convective_flux_op = Teuchos::rcp(
        new ClosureModel::IncompressibleConvectiveFlux<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            *ir, fluid_prop, user_data, "BOUNDARY_", "BOUNDARY_"));
    this->template registerEvaluator<EvalType>(fm, convective_flux_op);

    for (auto& pair : _equ_dof_ns_pair)
    {
        if (_continuity_model == ConModel::EDACTempNC
            && std::string::npos != pair.first.find("energy"))
        {
            continue;
        }
        this->registerConvectionTypeFluxOperator(
            pair, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);
    }

    // Create boundary cross-product flux for EP equation
    if (_build_ind_less_equ)
    {
        // Cross product term
        auto cross_product_flux_op = Teuchos::rcp(
            new ClosureModel::ElectricPotentialCrossProductFlux<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
                *ir, "BOUNDARY_", "BOUNDARY_"));
        this->template registerEvaluator<EvalType>(fm, cross_product_flux_op);

        // External magnetic field
        auto ext_magn_field_op = Teuchos::rcp(
            new ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>(
                *ir, user_data));
        this->template registerEvaluator<EvalType>(fm, ext_magn_field_op);

        for (auto& pair : _equ_dof_ep_pair)
        {
            this->registerConvectionTypeFluxOperator(
                pair, eq_vct_map, "ELECTRIC_POTENTIAL", fm, side_pb, 1.0);
        }
    }

    // Second-order flux //

    // NS equations
    if (_build_viscous_flux)
    {
        // Create list of equations for use with viscous flux
        std::unordered_map<std::string, std::string> _equ_dof_ns_visc_pair
            = _equ_dof_ns_pair;

        // Check if the BC is a wall function type
        const Teuchos::ParameterList ns_bc_params
            = bc_params.isSublist("Navier-Stokes")
                  ? bc_params.sublist("Navier-Stokes")
                  : bc_params;
        const std::string bc_type = ns_bc_params.get<std::string>("Type");

        const bool is_wall_func
            = std::string::npos != bc_type.find("Wall Function") ? true : false;

        // Add shear stress residual for wall function boundaries
        if (is_wall_func)
        {
            // Remove momentum equations from viscous equation set so that
            // the penalty method residual is not built in addition to the
            // wall function residual
            for (auto eqn = _equ_dof_ns_visc_pair.begin();
                 eqn != _equ_dof_ns_visc_pair.end();)
            {
                if (std::string::npos != eqn->first.find("momentum"))
                {
                    eqn = _equ_dof_ns_visc_pair.erase(eqn);
                }
                else
                {
                    ++eqn;
                }
            }

            // Create evaluator for wall function shear stress components
            const std::string wall_func_stress_prefix = "wall_func_stress_";

            const auto wall_func_stress_op = Teuchos::rcp(
                new BoundaryCondition::IncompressibleWallFunctionStress<
                    EvalType,
                    panzer::Traits,
                    num_space_dim>(*ir, wall_func_stress_prefix));

            this->template registerEvaluator<EvalType>(fm, wall_func_stress_op);

            // Create wall shear stress integrals
            const auto add_wall_stress_op = [&, this](
                                                const std::string& eqn_name,
                                                const std::string& dof_name) {
                // Construct basis
                const auto basis_layout
                    = this->getBasisIRLayout(side_pb, dof_name);

                // Shear stress op
                const std::string stress_residual_name
                    = "WALL_FUNC_STRESS_RESIDUAL_" + eqn_name;

                const auto shear_stress_op = Teuchos::rcp(
                    new panzer::Integrator_BasisTimesScalar<EvalType,
                                                            panzer::Traits>(
                        panzer::EvaluatorStyle::EVALUATES,
                        stress_residual_name,
                        wall_func_stress_prefix + eqn_name,
                        *basis_layout,
                        *ir,
                        1.0));

                this->template registerEvaluator<EvalType>(fm, shear_stress_op);
            };

            // Create shear stress residual for all velocity components
            for (auto& pair : _equ_dof_ns_pair)
            {
                if (std::string::npos != pair.first.find("momentum"))
                {
                    add_wall_stress_op(pair.first, pair.second);
                    eq_vct_map[pair.first].push_back(
                        "WALL_FUNC_STRESS_RESIDUAL_" + pair.first);
                }
            }
        }

        // Register penalty and viscous gradient operators for each equation.
        for (auto& pair : _equ_dof_ns_pair)
        {
            this->registerPenaltyAndViscousGradientOperator(
                pair, fm, side_pb, bc_params);
        }

        // Create boundary fluxes to be used with the penalty method
        for (auto& pair : this->bnd_prefix)
        {
            // Prefix names
            const std::string flux_prefix = pair.first;
            const std::string gradient_prefix = pair.second;

            auto viscous_flux_op = Teuchos::rcp(
                new ClosureModel::IncompressibleViscousFlux<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
                    *ir,
                    fluid_prop,
                    user_data,
                    _turbulence_model,
                    flux_prefix,
                    gradient_prefix));
            this->template registerEvaluator<EvalType>(fm, viscous_flux_op);

            if (std::string::npos != stabilization_method.find("Grad-Div"))
            {
                auto div_grad_op = Teuchos::rcp(
                    new ClosureModel::IncompressibleGradDiv<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
                        *ir, stability_param_list, flux_prefix, gradient_prefix));
                this->template registerEvaluator<EvalType>(fm, div_grad_op);
            }
        }

        // Create viscous flux integrals.
        for (auto& pair : _equ_dof_ns_visc_pair)
        {
            this->registerViscousTypeFluxOperator(
                pair, eq_vct_map, "VISCOUS", fm, side_pb, 1.0);
        }
    }

    // EP equation
    if (_build_ind_less_equ)
    {
        // Register penalty and viscous gradient operators for each equation.
        for (auto& pair : _equ_dof_ep_pair)
        {
            this->registerPenaltyAndViscousGradientOperator(
                pair, fm, side_pb, bc_params);
        }

        // Create boundary fluxes to be used with the penalty method
        for (auto& pair : this->bnd_prefix)
        {
            // Prefix names
            const std::string flux_prefix = pair.first;
            const std::string gradient_prefix = pair.second;

            auto diffusion_flux_op = Teuchos::rcp(
                new ClosureModel::ElectricPotentialDiffusionFlux<EvalType,
                                                                 panzer::Traits>(
                    *ir, flux_prefix, gradient_prefix));
            this->template registerEvaluator<EvalType>(fm, diffusion_flux_op);
        }

        // Create diffusion flux integral
        for (auto& pair : _equ_dof_ep_pair)
        {
            this->registerViscousTypeFluxOperator(
                pair, eq_vct_map, "ELECTRIC_POTENTIAL", fm, side_pb, 1.0);
        }
    }

    // Turbulence model boundary fluxes //

    // TM equation: create first-order and second-order boundary fluxes
    if (_turbulence_model)
    {
        // Use correct parameter list based on model equation count
        const auto turb_bc_params
            = _equ_dof_tm_pair.empty() ? Teuchos::ParameterList()
                                       : bc_params.sublist("Turbulence Model");

        // Define turbulent eddy viscosity on boundaries for wall functions
        for (auto& pair : this->bnd_prefix)
        {
            // Prefix names
            const std::string flux_prefix = pair.first;

            // Create evaluator for boundary turbulent eddy viscosity
            auto eddy_visc_op = Teuchos::rcp(
                new TurbulenceBoundaryEddyViscosity<EvalType, panzer::Traits>(
                    *ir, turb_bc_params, flux_prefix));

            this->template registerEvaluator<EvalType>(fm, eddy_visc_op);
        }

        // Register diffusivity coefficients and boundary conditions for each
        // turbulence model equation
        const auto tm_boundary_state_op = TurbulenceBoundaryStateFactory<
            EvalType,
            panzer::Traits,
            num_space_dim>::create(*ir,
                                   turb_bc_params,
                                   user_data,
                                   _turbulence_model_name);

        for (std::size_t i = 0; i < tm_boundary_state_op.size(); ++i)
        {
            this->template registerEvaluator<EvalType>(
                fm, tm_boundary_state_op[i]);
        }

        // Loop over each pair of the turbulence model equation(s)
        for (auto& pair_tm : _equ_dof_tm_pair)
        {
            Teuchos::ParameterList tm_name_list;
            tm_name_list.set("Field Name", pair_tm.second);
            tm_name_list.set("Equation Name", pair_tm.first);

            // Create boundary convective flux for each equation
            const auto convective_flux_op = Teuchos::rcp(
                new ClosureModel::VariableConvectiveFlux<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>(
                    *ir, tm_name_list, "BOUNDARY_", "BOUNDARY_"));
            this->template registerEvaluator<EvalType>(fm, convective_flux_op);

            BoundaryFluxBase<EvalType, NumSpaceDim>::registerConvectionTypeFluxOperator(
                pair_tm, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);

            // Register penalty and viscous gradient operators for each
            // equation.
            BoundaryFluxBase<EvalType, NumSpaceDim>::
                registerPenaltyAndViscousGradientOperator(
                    pair_tm, fm, side_pb, bc_params);

            // Create boundary fluxes to be used with the penalty method
            for (auto& pair_bnd :
                 BoundaryFluxBase<EvalType, NumSpaceDim>::bnd_prefix)
            {
                const std::string flux_prefix = pair_bnd.first;
                const std::string gradient_prefix = pair_bnd.second;

                const auto diffusion_flux_op = Teuchos::rcp(
                    new ClosureModel::VariableDiffusionFlux<EvalType,
                                                            panzer::Traits>(
                        *ir, tm_name_list, flux_prefix, gradient_prefix));
                this->template registerEvaluator<EvalType>(fm,
                                                           diffusion_flux_op);
            }

            // Create diffusion flux integral
            BoundaryFluxBase<EvalType, NumSpaceDim>::registerViscousTypeFluxOperator(
                pair_tm, eq_vct_map, "DIFFUSION", fm, side_pb, 1.0);
        }
    }

    // Compose total residual for NS equations
    for (auto& pair : _equ_dof_ns_pair)
    {
        this->registerResidual(pair, eq_vct_map, fm, side_pb);
    }

    // Compose total residual for EP equation
    for (auto& pair : _equ_dof_ep_pair)
    {
        this->registerResidual(pair, eq_vct_map, fm, side_pb);
    }

    // Compose total residual for TM equation
    for (auto& pair : _equ_dof_tm_pair)
    {
        BoundaryFluxBase<EvalType, NumSpaceDim>::registerResidual(
            pair, eq_vct_map, fm, side_pb);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::
    buildAndRegisterScatterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& /*user_data*/) const
{
    for (auto& pair : _equ_dof_ns_pair)
    {
        this->registerScatterOperator(pair, fm, side_pb, lof);
    }

    for (auto& pair : _equ_dof_ep_pair)
    {
        this->registerScatterOperator(pair, fm, side_pb, lof);
    }

    for (auto& pair : _equ_dof_tm_pair)
    {
        BoundaryFluxBase<EvalType, NumSpaceDim>::registerScatterOperator(
            pair, fm, side_pb, lof);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::
    buildAndRegisterGatherAndOrientationEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& user_data) const
{
    side_pb.buildAndRegisterGatherAndOrientationEvaluators(fm, lof, user_data);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::postRegistrationSetup(
    typename panzer::Traits::SetupData, PHX::FieldManager<panzer::Traits>&)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::evaluateFields(
    typename panzer::Traits::EvalData)
{
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_IMPL_HPP
