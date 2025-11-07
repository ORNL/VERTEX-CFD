#ifndef VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLELSVOFBOUNDARYFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLELSVOFBOUNDARYFLUX_IMPL_HPP

#include "VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include "incompressible_lsvof_solver/boundary_conditions/VertexCFD_IncompressibleLSVOFBoundaryState_Factory.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFConvectiveFlux.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFNthPhaseFraction.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarConvectiveFlux.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFVariableProperties.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFViscousFlux.hpp"

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

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
IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::IncompressibleLSVOFBoundaryFlux(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : BoundaryFluxBase<EvalType, NumSpaceDim>(bc, global_data)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::setup(
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& user_data)
{
    // Check if momentum equations should be built
    _build_mom_eqn = user_data.isType<bool>("Build LSVOF Momentum Equation")
                         ? user_data.get<bool>("Build LSVOF Momentum Equation")
                         : true;

    // Initialize NS equation and DOF names
    if (_build_mom_eqn)
    {
        _equ_dof_ns_pair.insert({"continuity", "lagrange_pressure"});

        for (int d = 0; d < num_space_dim; ++d)
        {
            const std::string ds = std::to_string(d);
            _equ_dof_ns_pair.insert({"momentum_" + ds, "velocity_" + ds});
        }
    }

    // Get physics params from physics block
    const auto& physics_params = side_pb.getParameterList()->sublist("child0");

    // Get LSVOF model info from parameter list
    _lsvof_params = physics_params.sublist("LSVOF_Properties");

    const std::string lsvof_model_name
        = _lsvof_params.get<std::string>("LSVOF Model");

    const int num_phases = _lsvof_params.get<int>("Number of Phases");

    // Initialize LSVOF equation and DOF names
    if (lsvof_model_name == "VOF")
    {
        for (int n = 1; n <= num_phases; ++n)
        {
            const std::string list_name = "Phase " + std::to_string(n);

            Teuchos::ParameterList phase_list
                = _lsvof_params.sublist(list_name);

            const std::string phase_name
                = phase_list.get<std::string>("Phase Name");

            const std::string phase_fraction_name = "alpha_" + phase_name;

            _all_phase_names.push_back(phase_fraction_name);

            if (n < num_phases)
            {
                _equ_dof_lsvof_pair.insert(
                    {phase_fraction_name + "_equation", phase_fraction_name});

                _dof_phase_names.push_back(phase_fraction_name);
            }
        }
    }

    // Initialize parent class variables (only needed with one set of
    // equations)
    this->initialize(side_pb, _equ_dof_lsvof_pair);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::buildAndRegisterEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&,
    const Teuchos::ParameterList&,
    const Teuchos::ParameterList& user_data) const
{
    // Map to store residuals for each equation listed in pairs
    std::unordered_map<std::string, std::vector<std::string>> eq_vct_map;

    // Get integration rule for closure models
    const auto ir = this->integrationRule();

    // Create degree of freedom and gradients for NS equations
    for (auto& pair : _equ_dof_ns_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Create degree of freedom and gradients for LSVOF equations
    for (auto& pair : _equ_dof_lsvof_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Create LSVOF scalar field array
    const bool include_time_deriv = false;
    const bool include_grads = false;

    const auto phase_vec_op
        = Utils::ScalarToVector<EvalType, PhaseIndex>::createFromList(
            *ir,
            "volume_fractions",
            _dof_phase_names,
            include_time_deriv,
            include_grads);

    this->template registerEvaluator<EvalType>(fm, phase_vec_op);

    // Create boundary Nth phase fraction
    const auto nth_phase_op = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFNthPhaseFraction<EvalType,
                                                              panzer::Traits>(
            *ir, _all_phase_names));

    this->template registerEvaluator<EvalType>(fm, nth_phase_op);

    // Register normals
    this->registerSideNormals(fm, side_pb);

    // Create boundary state operators
    const auto bc_params = *(this->m_bc.params());

    auto incomp_lsvof_boundary_state_op
        = IncompressibleLSVOFBoundaryStateFactory<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>::
            create(*ir, _all_phase_names, bc_params, _lsvof_params, user_data);
    this->template registerEvaluator<EvalType>(fm,
                                               incomp_lsvof_boundary_state_op);

    // Create fluid property fields
    auto var_prop_op = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFVariableProperties<EvalType,
                                                                panzer::Traits>(
            *ir, _lsvof_params, _all_phase_names, false, "BOUNDARY_"));

    this->template registerEvaluator<EvalType>(fm, var_prop_op);

    // First-order flux //

    // Create boundary convective fluxes for NS equations
    if (_build_mom_eqn)
    {
        auto convective_flux_op = Teuchos::rcp(
            new ClosureModel::IncompressibleLSVOFConvectiveFlux<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
                *ir, "BOUNDARY_", "BOUNDARY_"));
        this->template registerEvaluator<EvalType>(fm, convective_flux_op);

        for (auto& pair : _equ_dof_ns_pair)
        {
            this->registerConvectionTypeFluxOperator(
                pair, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);
        }
    }

    // Create boundary convective fluxes for LSVOF equations
    int phase_index = 0;
    for (auto& pair : _equ_dof_lsvof_pair)
    {
        const auto& equ_name = pair.first;
        const auto& dof_name = pair.second;

        Teuchos::ParameterList conv_flux_params;
        conv_flux_params.set("Field Name", dof_name);
        conv_flux_params.set("Equation Name", equ_name);

        auto lsvof_convective_flux_op = Teuchos::rcp(
            new ClosureModel::IncompressibleLSVOFScalarConvectiveFlux<
                EvalType,
                panzer::Traits,
                num_space_dim>(*ir,
                               phase_index,
                               _dof_phase_names.size(),
                               equ_name,
                               "BOUNDARY_",
                               "BOUNDARY_"));

        this->template registerEvaluator<EvalType>(fm,
                                                   lsvof_convective_flux_op);

        this->registerConvectionTypeFluxOperator(
            pair, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);

        ++phase_index;
    }

    // Second-order flux //

    // Register penalty and viscous gradient operators for each equation.
    if (_build_mom_eqn)
    {
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
            const std::string field_prefix = "BOUNDARY_";

            auto viscous_flux_op = Teuchos::rcp(
                new ClosureModel::IncompressibleLSVOFViscousFlux<EvalType,
                                                                 panzer::Traits,
                                                                 num_space_dim>(
                    *ir,
                    _lsvof_params,
                    user_data,
                    flux_prefix,
                    gradient_prefix,
                    field_prefix));
            this->template registerEvaluator<EvalType>(fm, viscous_flux_op);
        }

        // Create viscous flux integrals.
        for (auto& pair : _equ_dof_ns_pair)
        {
            this->registerViscousTypeFluxOperator(
                pair, eq_vct_map, "VISCOUS", fm, side_pb, 1.0);
        }

        // Compose total residual for NS equations
        for (auto& pair : _equ_dof_ns_pair)
        {
            this->registerResidual(pair, eq_vct_map, fm, side_pb);
        }
    }

    // Compose total residual for LSVOF equations
    for (auto& pair : _equ_dof_lsvof_pair)
    {
        this->registerResidual(pair, eq_vct_map, fm, side_pb);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::
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

    for (auto& pair : _equ_dof_lsvof_pair)
    {
        this->registerScatterOperator(pair, fm, side_pb, lof);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::
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
void IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::postRegistrationSetup(
    typename panzer::Traits::SetupData, PHX::FieldManager<panzer::Traits>&)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFBoundaryFlux<EvalType, NumSpaceDim>::evaluateFields(
    typename panzer::Traits::EvalData)
{
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_BOUNDARYCONDITION_INCOMPRESSIBLELSVOFBOUNDARYFLUX_IMPL_HPP
