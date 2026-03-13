#ifndef VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONBOUNDARYFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONBOUNDARYFLUX_IMPL_HPP

#include "boundary_conditions/VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "boundary_conditions/VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"

#include "conduction/boundary_conditions/VertexCFD_ConductionBoundaryState_Factory.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionFlux.hpp"

#include <Panzer_DOF.hpp>
#include <Panzer_DOFGradient.hpp>
#include <Panzer_DotProduct.hpp>
#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Normals.hpp>
#include <Panzer_Sum.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_MDField.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
ConductionBoundaryFlux<EvalType, NumSpaceDim>::ConductionBoundaryFlux(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : BoundaryFluxBase<EvalType, NumSpaceDim>(bc, global_data)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void ConductionBoundaryFlux<EvalType, NumSpaceDim>::setup(
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& /**user_data**/)
{
    // Initialize equation names and variable names for conduction equation
    _equ_dof_cond_pair.insert({"energy", "temperature"});

    // Initialize parent class variables (only needed with one set of
    // equations)
    this->initialize(side_pb, _equ_dof_cond_pair);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void ConductionBoundaryFlux<EvalType, NumSpaceDim>::buildAndRegisterEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>&,
    const Teuchos::ParameterList& models,
    const Teuchos::ParameterList& /*user_data*/) const
{
    // Create boundary state operators for conduction equation
    // Get bc sublist
    const Teuchos::ParameterList bc_params = *(this->m_bc.params());

    // Get model id from boundary conditions or physics block
    std::string model_id = "";
    Teuchos::ParameterList side_pb_list;
    this->getModelID(bc_params, side_pb, model_id, side_pb_list);

    // Map to store residuals for each equation listed in `_equ_dof_cond_pair`
    std::unordered_map<std::string, std::vector<std::string>> eq_vct_map;

    // Get integration rule for closure models
    const auto ir = this->integrationRule();

    // Create degree of freedom and gradients for conduction equation
    for (auto& pair : _equ_dof_cond_pair)
    {
        this->registerDOFsGradient(fm, side_pb, pair.second);
    }

    // Register normals
    this->registerSideNormals(fm, side_pb);

    // Register conduction boundary condition and closure models
    const auto closure_model = models.sublist(model_id);
    const auto boundary_state_op
        = ConductionBoundaryStateFactory<EvalType,
                                         panzer::Traits,
                                         num_space_dim>::create(*ir,
                                                                bc_params,
                                                                closure_model);

    for (std::size_t i = 0; i < boundary_state_op.size(); ++i)
    {
        this->template registerEvaluator<EvalType>(fm, boundary_state_op[i]);
    }

    // Second-order flux //

    // Register penalty and viscous gradient operators for each equation.
    for (auto& pair : _equ_dof_cond_pair)
    {
        this->registerPenaltyAndViscousGradientOperator(
            pair, fm, side_pb, bc_params);
    }

    // Create boundary fluxes to be used with the penalty method
    for (const auto& [flux_prefix, gradient_prefix] : this->bnd_prefix)
    {
        auto viscous_flux_op = Teuchos::rcp(
            new ClosureModel::ConductionFlux<EvalType, panzer::Traits>(
                *ir, flux_prefix, gradient_prefix));
        this->template registerEvaluator<EvalType>(fm, viscous_flux_op);
    }

    // Create viscous flux integrals.
    for (auto& pair : _equ_dof_cond_pair)
    {
        this->registerViscousTypeFluxOperator(
            pair, eq_vct_map, "CONDUCTION", fm, side_pb, 1.0);
    }

    // Compose total residual for conduction equation
    for (auto& pair : _equ_dof_cond_pair)
    {
        this->registerResidual(pair, eq_vct_map, fm, side_pb);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void ConductionBoundaryFlux<EvalType, NumSpaceDim>::buildAndRegisterScatterEvaluators(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const panzer::LinearObjFactory<panzer::Traits>& lof,
    const Teuchos::ParameterList& /*user_data*/) const
{
    for (auto& pair : _equ_dof_cond_pair)
    {
        this->registerScatterOperator(pair, fm, side_pb, lof);
    }
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONBOUNDARYFLUX_IMPL_HPP
