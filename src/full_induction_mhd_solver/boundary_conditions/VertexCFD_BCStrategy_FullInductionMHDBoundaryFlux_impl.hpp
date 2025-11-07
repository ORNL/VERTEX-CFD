#ifndef VERTEXCFD_BOUNDARYCONDITION_FULLINDUCTIONMHDBOUNDARYFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_FULLINDUCTIONMHDBOUNDARYFLUX_IMPL_HPP

#include "boundary_conditions/VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "boundary_conditions/VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_IncompressibleBoundaryState_Factory.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousFlux.hpp"

#include "incompressible_solver/fluid_properties/VertexCFD_Closure_IncompressibleFluidProperties.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

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
FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::FullInductionMHDBoundaryFlux(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : BoundaryFluxBase<EvalType, NumSpaceDim>(bc, global_data)
{
    // Check if boundary is an internal interface between solid and fluid
    // regions
    _internal_interface = bc.params()->isType<bool>("Fluid/Solid Interface")
                              ? bc.params()->get<bool>("Fluid/Solid Interface")
                              : false;

    // For now, throw an error on interfaces as the logic for solid regions
    // does not exist yet
    if (_internal_interface)
    {
        throw std::runtime_error(
            "Fluid/Solid Interface not yet enabled for Full Induction MHD");
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::setup(
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& /*user_data*/)
{
    // Get the physics block parameters associated with this sideset,
    // and extract necessary information, including "Model IDs" for
    // the incompressible NS and full induction closure models
    const auto side_pb_params = *side_pb.getParameterList();
    bool build_magn_corr = false;
    for (int i = 0; i < side_pb_params.numParams(); ++i)
    {
        const std::string name = "child" + std::to_string(i);
        if (side_pb_params.isSublist(name))
        {
            const auto& sl = side_pb_params.sublist(name);
            const auto& type = sl.get<std::string>("Type");
            if (type == "FullInductionMHD")
            {
                _induction_model_id = sl.get<std::string>("Model ID");
                if (sl.isType<bool>("Build Magnetic Correction Potential "
                                    "Equation"))
                {
                    build_magn_corr = sl.get<bool>(
                        "Build Magnetic Correction Potential Equation");
                }
            }
            else if (type == "IncompressibleNavierStokes")
            {
                _incompressible_model_id = sl.get<std::string>("Model ID");
                _build_viscous_flux = sl.get<bool>("Build Viscous Flux");
            }
        }
    }

    // Internal interface case: magnetic field and correction potential should
    // not require boundary condition flux.
    _build_full_induction_model = !(_internal_interface);

    // Initialize equation names and variable names for NS equations.
    _equ_dof_ns_pair.insert({"continuity", "lagrange_pressure"});
    for (int d = 0; d < num_space_dim; ++d)
    {
        const std::string ds = std::to_string(d);
        _equ_dof_ns_pair.insert({"momentum_" + ds, "velocity_" + ds});
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
void FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::buildAndRegisterEvaluators(
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

    // Create degree of freedom and gradients for TM equations
    for (auto& pair : _equ_dof_fim_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Register normals
    this->registerSideNormals(fm, side_pb);

    // Fluid properties: model id is stored in a sublist
    Teuchos::ParameterList fluid_prop_list
        = closure_models.sublist(_incompressible_model_id)
              .sublist("Fluid Properties");

    fluid_prop_list.set<bool>("Build Temperature Equation", false);
    fluid_prop_list.set<bool>("Build Buoyancy Source", false);
    fluid_prop_list.set<bool>("Build Inductionless MHD Equation", false);

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
        this->registerConvectionTypeFluxOperator(
            pair, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);
    }

    // Second-order flux //

    // NS equations
    if (_build_viscous_flux)
    {
        // Check if the BC is a wall function type
        const Teuchos::ParameterList ns_bc_params
            = bc_params.isSublist("Navier-Stokes")
                  ? bc_params.sublist("Navier-Stokes")
                  : bc_params;
        const std::string bc_type = ns_bc_params.get<std::string>("Type");

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

            const bool turb_model = false;
            auto viscous_flux_op = Teuchos::rcp(
                new ClosureModel::IncompressibleViscousFlux<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
                    *ir,
                    fluid_prop,
                    user_data,
                    turb_model,
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

    // Full Induction Model boundary fluxes
    if (_build_full_induction_model)
    {
        const auto full_induction_params
            = closure_models.sublist(_induction_model_id)
                  .sublist("Full Induction MHD Properties");
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
                        pair_fim, fm, side_pb, bc_params);

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

    // Compose total residual for FIM equations
    for (auto& pair : _equ_dof_fim_pair)
    {
        BoundaryFluxBase<EvalType, NumSpaceDim>::registerResidual(
            pair, eq_vct_map, fm, side_pb);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::
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

    for (auto& pair : _equ_dof_fim_pair)
    {
        BoundaryFluxBase<EvalType, NumSpaceDim>::registerScatterOperator(
            pair, fm, side_pb, lof);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::
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
void FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::postRegistrationSetup(
    typename panzer::Traits::SetupData, PHX::FieldManager<panzer::Traits>&)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void FullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::evaluateFields(
    typename panzer::Traits::EvalData)
{
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_FULLINDUCTIONMHDBOUNDARYFLUX_IMPL_HPP
