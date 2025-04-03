#ifndef VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLEBOUNDARYFLUX_IMPL_HPP

#include "VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include "closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_IncompressibleBoundaryState_Factory.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousFlux.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_ElectricPotentialBoundaryState_Factory.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialCrossProductFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_ElectricPotentialDiffusionFlux.hpp"

#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceBoundaryEddyViscosity.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_TurbulenceBoundaryState_Factory.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleVariableConvectiveFlux.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleVariableDiffusionFlux.hpp"

#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_FullInductionBoundaryState_Factory.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConvectiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionResistiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticFieldGradient.hpp"
#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

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
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleBoundaryFlux<EvalType, NumSpaceDim>::setup(
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& user_data)
{
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

    // Turbulence model boolean
    _turbulence_model_name
        = user_data.isType<std::string>("Turbulence Model")
              ? user_data.get<std::string>("Turbulence Model")
              : "No Turbulence Model";
    _turbulence_model = _turbulence_model_name == "No Turbulence Model" ? false
                                                                        : true;

    // Full induction model boolean
    _build_full_induction_model
        = user_data.isType<bool>("Build Full Induction Model")
              ? user_data.get<bool>("Build Full Induction Model")
              : false;
    bool build_magn_corr = false;
    if (_build_full_induction_model)
    {
        const auto mhd_prop_list
            = user_data.sublist("Full Induction MHD Properties");
        if (mhd_prop_list.isType<bool>("Build Magnetic Correction Potential "
                                       "Equation"))
        {
            build_magn_corr = mhd_prop_list.get<bool>(
                "Build Magnetic Correction Potential Equation");
        }
    }

    // Initialize equation names and variable names for NS equations
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
    }

    // Initialize equation names and variables for FIM
    if (_build_full_induction_model)
    {
        for (int d = 0; d < num_space_dim; ++d)
        {
            const std::string ds = std::to_string(d);
            _equ_dof_fim_pair.insert(
                {"induction_" + ds, "induced_magnetic_field_" + ds});
        }
        if (build_magn_corr)
        {
            _equ_dof_fim_pair.insert({"magnetic_correction_potential",
                                      "scalar_magnetic_potential"});
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
    const Teuchos::ParameterList&,
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

    // Create degree of freedom and gradients for TM equations
    for (auto& pair : _equ_dof_fim_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Register normals
    this->registerSideNormals(fm, side_pb);

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
                                                                    user_data);
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

    // Fluid properties
    Teuchos::ParameterList fluid_prop_list
        = user_data.sublist("Fluid Properties");
    fluid_prop_list.set<bool>("Build Temperature Equation", _build_temp_equ);
    fluid_prop_list.set<bool>("Build Inductionless MHD Equation",
                              _build_ind_less_equ);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    // First-order flux //

    // Create boundary convective fluxes for NS equations
    auto convective_flux_op = Teuchos::rcp(
        new ClosureModel::IncompressibleConvectiveFlux<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            *ir, fluid_prop, "BOUNDARY_", "BOUNDARY_"));
    this->template registerEvaluator<EvalType>(fm, convective_flux_op);

    for (auto& pair : _equ_dof_ns_pair)
    {
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
                *ir, fluid_prop, "BOUNDARY_", "BOUNDARY_"));
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
        // Register penalty and viscous gradient operators for each equation.
        for (auto& pair : _equ_dof_ns_pair)
        {
            this->registerPenaltyAndViscousGradientOperator(
                pair, fm, side_pb, user_data);
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
        }

        // Create viscous flux integrals.
        for (auto& pair : _equ_dof_ns_pair)
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
                pair, fm, side_pb, user_data);
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
                    *ir, fluid_prop, flux_prefix, gradient_prefix));
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
                                   _turbulence_model_name,
                                   fluid_prop);

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
                new ClosureModel::IncompressibleVariableConvectiveFlux<
                    EvalType,
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
                    pair_tm, fm, side_pb, user_data);

            // Create boundary fluxes to be used with the penalty method
            for (auto& pair_bnd :
                 BoundaryFluxBase<EvalType, NumSpaceDim>::bnd_prefix)
            {
                const std::string flux_prefix = pair_bnd.first;
                const std::string gradient_prefix = pair_bnd.second;

                const auto diffusion_flux_op = Teuchos::rcp(
                    new ClosureModel::IncompressibleVariableDiffusionFlux<
                        EvalType,
                        panzer::Traits,
                        NumSpaceDim>(
                        *ir, tm_name_list, flux_prefix, gradient_prefix));
                this->template registerEvaluator<EvalType>(fm,
                                                           diffusion_flux_op);
            }

            // Create diffusion flux integral
            BoundaryFluxBase<EvalType, NumSpaceDim>::registerViscousTypeFluxOperator(
                pair_tm, eq_vct_map, "DIFFUSION", fm, side_pb, 1.0);
        }
    }

    // Full Induction Model boundary fluxes

    if (_build_full_induction_model)
    {
        const auto full_induction_params
            = user_data.sublist("Full Induction MHD Properties");
        const MHDProperties::FullInductionMHDProperties mhd_props(
            full_induction_params);

        // Register boundary conditions for the induciton equations
        const auto fim_boundary_state_op = FullInductionBoundaryStateFactory<
            EvalType,
            panzer::Traits,
            num_space_dim>::create(*ir,
                                   bc_params.sublist("Full Induction Model"),
                                   user_data,
                                   mhd_props);

        for (std::size_t i = 0; i < fim_boundary_state_op.size(); ++i)
        {
            this->template registerEvaluator<EvalType>(
                fm, fim_boundary_state_op[i]);
        }

        auto induction_flux_op = Teuchos::rcp(
            new ClosureModel::InductionConvectiveFlux<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>(
                *ir, mhd_props, "BOUNDARY_", "BOUNDARY_"));
        this->template registerEvaluator<EvalType>(fm, induction_flux_op);

        for (auto& pair_fim : _equ_dof_fim_pair)
        {
            BoundaryFluxBase<EvalType, NumSpaceDim>::registerConvectionTypeFluxOperator(
                pair_fim, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);
        }

        if (mhd_props.buildResistiveFlux())
        {
            for (auto& pair_fim : _equ_dof_fim_pair)
            {
                // Register penalty and resistive gradient operators for each
                // equation.
                BoundaryFluxBase<EvalType, NumSpaceDim>::
                    registerPenaltyAndViscousGradientOperator(
                        pair_fim, fm, side_pb, user_data);

                // Create boundary fluxes to be used with the penalty method
                for (auto& pair_bnd :
                     BoundaryFluxBase<EvalType, NumSpaceDim>::bnd_prefix)
                {
                    const std::string flux_prefix = pair_bnd.first;
                    const std::string gradient_prefix = pair_bnd.second;

                    // Need total magnetic field symmetry and penalty gradients
                    const auto tot_magn_field_grad_op = Teuchos::rcp(
                        new ClosureModel::TotalMagneticFieldGradient<
                            EvalType,
                            panzer::Traits,
                            NumSpaceDim>(*ir, gradient_prefix));
                    this->template registerEvaluator<EvalType>(
                        fm, tot_magn_field_grad_op);

                    const auto resistive_flux_op = Teuchos::rcp(
                        new ClosureModel::InductionResistiveFlux<EvalType,
                                                                 panzer::Traits,
                                                                 NumSpaceDim>(
                            *ir, mhd_props, flux_prefix, gradient_prefix));
                    this->template registerEvaluator<EvalType>(
                        fm, resistive_flux_op);
                }
                // Create resistive flux integral
                BoundaryFluxBase<EvalType, NumSpaceDim>::registerViscousTypeFluxOperator(
                    pair_fim, eq_vct_map, "RESISTIVE", fm, side_pb, 1.0);
            }
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

    // Compose total residual for FIM equations
    for (auto& pair : _equ_dof_fim_pair)
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

    for (auto& pair : _equ_dof_fim_pair)
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
