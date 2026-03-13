#ifndef VERTEXCFD_BOUNDARYCONDITION_SOLIDFULLINDUCTIONMHDBOUNDARYFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_SOLIDFULLINDUCTIONMHDBOUNDARYFLUX_IMPL_HPP

#include "boundary_conditions/VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "boundary_conditions/VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"
#include "closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"

#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_FullInductionBoundaryState_Factory.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionResistiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_SolidFullInductionConvectiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticField.hpp"
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
#include <stdexcept>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
SolidFullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::
    SolidFullInductionMHDBoundaryFlux(
        const panzer::BC& bc,
        const Teuchos::RCP<panzer::GlobalData>& global_data)
    : BoundaryFluxBase<EvalType, NumSpaceDim>(bc, global_data)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void SolidFullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::setup(
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& /*user_data*/)
{
    // Get the solid induction physics block parameters associated with this
    // sideset, and extract necessary information, including "Model IDs" for
    // induction closure models.
    const auto side_pb_params = *side_pb.getParameterList();
    bool build_magn_corr = false;
    bool found_pb = false;
    for (int i = 0; i < side_pb_params.numParams(); ++i)
    {
        const std::string name = "child" + std::to_string(i);
        if (side_pb_params.isSublist(name))
        {
            const auto& sl = side_pb_params.sublist(name);
            const auto& type = sl.get<std::string>("Type");
            if (type == "SolidFullInductionMHD")
            {
                _induction_model_id = sl.get<std::string>("Model ID");
                if (sl.isType<bool>("Build Magnetic Correction Potential "
                                    "Equation"))
                {
                    build_magn_corr = sl.get<bool>(
                        "Build Magnetic Correction Potential Equation");
                }
                found_pb = true;
            }
        }
    }
    _build_convective_flux = build_magn_corr;
    if (!found_pb)
    {
        const std::string ss_id = this->m_bc.sidesetID();
        throw std::runtime_error(
            "No \"SolidFullInduction\" physics block found for side set "
            + this->m_bc.sidesetID());
    }

    // Initialize equation names and variables for FIM
    for (int d = 0; d < num_space_dim; ++d)
    {
        const std::string ds = std::to_string(d);
        _equ_dof_sim_pair.insert(
            {"induction_" + ds, "induced_magnetic_field_" + ds});
    }
    if (build_magn_corr)
    {
        _equ_dof_sim_pair.insert(
            {"magnetic_correction_potential", "scalar_magnetic_potential"});
    }

    // Initialize parent class variables (only needed with one set of
    // equations)
    this->initialize(side_pb, _equ_dof_sim_pair);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void SolidFullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::
    buildAndRegisterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&,
        const Teuchos::ParameterList& closure_models,
        const Teuchos::ParameterList& user_data) const
{
    // Map to store residuals for each equation listed in `_equ_dof_sim_pair`
    std::unordered_map<std::string, std::vector<std::string>> eq_vct_map;

    // Get integration rule for closure models
    const auto& ir = this->integrationRule();

    // Create degree of freedom and gradients for TM equations
    for (auto& pair : _equ_dof_sim_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Register normals
    this->registerSideNormals(fm, side_pb);

    // Create boundary state operators for NS equations and EP equation
    // Get bc sublist
    const auto bc_params = *(this->m_bc.params());

    // Full Induction Model boundary fluxes
    const auto full_induction_params
        = closure_models.sublist(_induction_model_id)
              .sublist("Full Induction MHD Properties");
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    // Add total magnetic field closure.
    const auto ext_magn_field_op = Teuchos::rcp(
        new ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>(
            *ir, user_data.sublist("External Magnetic Field Parameters")));
    this->template registerEvaluator<EvalType>(fm, ext_magn_field_op);

    const auto tot_magn_field_op = Teuchos::rcp(
        new ClosureModel::TotalMagneticField<EvalType, panzer::Traits, num_space_dim>(
            *ir, "BOUNDARY_"));
    this->template registerEvaluator<EvalType>(fm, tot_magn_field_op);

    // Register boundary conditions for the induciton equations
    const auto fim_boundary_state_op
        = FullInductionBoundaryStateFactory<EvalType, panzer::Traits, num_space_dim>::
            create(*ir, bc_params.sublist("Full Induction Model"), mhd_props);
    this->template registerEvaluator<EvalType>(fm, fim_boundary_state_op);

    if (_build_convective_flux)
    {
        const auto induction_flux_op = Teuchos::rcp(
            new ClosureModel::SolidFullInductionConvectiveFlux<EvalType,
                                                               panzer::Traits,
                                                               num_space_dim>(
                *ir, mhd_props, "BOUNDARY_", "BOUNDARY_"));
        this->template registerEvaluator<EvalType>(fm, induction_flux_op);

        for (auto& pair_fim : _equ_dof_sim_pair)
        {
            BoundaryFluxBase<EvalType, NumSpaceDim>::registerConvectionTypeFluxOperator(
                pair_fim, eq_vct_map, "CONVECTIVE", fm, side_pb, 1.0);
        }
    }

    if (mhd_props.buildResistiveFlux())
    {
        const auto resistivity_op = Teuchos::rcp(
            new ClosureModel::ConstantScalarField<EvalType, panzer::Traits>(
                *ir, "resistivity", mhd_props.resistivity()));
        this->template registerEvaluator<EvalType>(fm, resistivity_op);

        for (auto& pair_fim : _equ_dof_sim_pair)
        {
            // Register penalty and resistive gradient operators for each
            // equation.
            BoundaryFluxBase<EvalType, NumSpaceDim>::
                registerPenaltyAndViscousGradientOperator(
                    pair_fim, fm, side_pb, bc_params);

            // Create boundary fluxes to be used with the penalty method
            for (const auto& [flux_prefix, gradient_prefix] : this->bnd_prefix)
            {
                // Need total magnetic field symmetry and penalty gradients
                const auto tot_magn_field_grad_op = Teuchos::rcp(
                    new ClosureModel::TotalMagneticFieldGradient<EvalType,
                                                                 panzer::Traits,
                                                                 NumSpaceDim>(
                        *ir, gradient_prefix));
                this->template registerEvaluator<EvalType>(
                    fm, tot_magn_field_grad_op);

                const auto resistive_flux_op = Teuchos::rcp(
                    new ClosureModel::InductionResistiveFlux<EvalType,
                                                             panzer::Traits,
                                                             NumSpaceDim>(
                        *ir, mhd_props, flux_prefix, gradient_prefix));
                this->template registerEvaluator<EvalType>(fm,
                                                           resistive_flux_op);
            }
            // Create resistive flux integral
            BoundaryFluxBase<EvalType, NumSpaceDim>::registerViscousTypeFluxOperator(
                pair_fim, eq_vct_map, "RESISTIVE", fm, side_pb, 1.0);
        }
    }

    // Compose total residual for FIM equations
    for (auto& pair : _equ_dof_sim_pair)
    {
        BoundaryFluxBase<EvalType, NumSpaceDim>::registerResidual(
            pair, eq_vct_map, fm, side_pb);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void SolidFullInductionMHDBoundaryFlux<EvalType, NumSpaceDim>::
    buildAndRegisterScatterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& /*user_data*/) const
{
    for (auto& pair : _equ_dof_sim_pair)
    {
        BoundaryFluxBase<EvalType, NumSpaceDim>::registerScatterOperator(
            pair, fm, side_pb, lof);
    }
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_BOUNDARYCONDITION_SOLIDFULLINDUCTIONMHDBOUNDARYFLUX_IMPL_HPP
