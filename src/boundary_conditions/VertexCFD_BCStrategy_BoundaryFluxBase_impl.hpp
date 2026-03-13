#ifndef VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_IMPL_HPP
#define VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_IMPL_HPP

#include "VertexCFD_BCStrategy_BoundaryFluxBase.hpp"

#include "VertexCFD_BoundaryState_ViscousGradient.hpp"
#include "VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp"
#include "VertexCFD_BoundaryState_ViscousPenaltyParameterMetricTensor.hpp"
#include "VertexCFD_Integrator_BoundaryGradBasisDotVector.hpp"
#include "closure_models/VertexCFD_Closure_MetricTensor.hpp"

#include <Panzer_DOF.hpp>
#include <Panzer_DOFGradient.hpp>
#include <Panzer_DotProduct.hpp>
#include <Panzer_Integrator_BasisTimesScalar.hpp>
#include <Panzer_Normals.hpp>
#include <Panzer_Sum.hpp>

#include <Phalanx_DataLayout_MDALayout.hpp>
#include <Phalanx_FieldTag_Tag.hpp>

#include <map>
#include <sstream>
#include <stdexcept>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
BoundaryFluxBase<EvalType, NumSpaceDim>::BoundaryFluxBase(
    const panzer::BC& bc, const Teuchos::RCP<panzer::GlobalData>& global_data)
    : panzer::BCStrategy<EvalType>(bc)
    , panzer::GlobalDataAcceptorDefaultImpl(global_data)
{
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::
    buildAndRegisterGatherAndOrientationEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& user_data) const
{
    side_pb.buildAndRegisterGatherAndOrientationEvaluators(fm, lof, user_data);
}

//---------------------------------------------------------------------------//
// Initialize class members
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::initialize(
    const panzer::PhysicsBlock& side_pb,
    std::unordered_map<std::string, std::string>& /**dof_eq_map**/)
{
    // Get the maximum of integration rule order
    const auto& irules = side_pb.getIntegrationRules();
    int integration_order = 0;
    for (const auto& pair : irules)
    {
        integration_order = std::max(integration_order, pair.second->order());
    }

    // Assign an integration rule for side normal evaluator
    _ir = Teuchos::rcp(
        new panzer::IntegrationRule(integration_order, side_pb.cellData()));

    // Create a map between dof and integration rule
    const auto& dof_basis_pair = side_pb.getProvidedDOFs();
    for (const auto& dof_basis : dof_basis_pair)
    {
        _dof_basis_pair.insert({dof_basis.first, dof_basis.second});
        const Teuchos::RCP<panzer::IntegrationRule> ir = Teuchos::rcp(
            new panzer::IntegrationRule(integration_order, side_pb.cellData()));
    }
}

//---------------------------------------------------------------------------//
// Get model id class members: this function retrieves the model id from the
// boundary conditions (multiple equation sets per block) or physics blocks
// (one equation set per block)
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::getModelID(
    const Teuchos::ParameterList& bc_params,
    const panzer::PhysicsBlock& side_pb,
    std::string& model_id,
    Teuchos::ParameterList& side_pb_list) const
{
    if (bc_params.isType<std::string>("Model ID"))
    {
        // TODO: This logic is currently not used and will have to be updated
        // when using composite boundaries.
        model_id = bc_params.get<std::string>("Model ID");
        for (auto itr = side_pb.getParameterList()->begin();
             itr != side_pb.getParameterList()->end();
             ++itr)
        {
            const auto sublist
                = Teuchos::getValue<Teuchos::ParameterList>(itr->second);
            if (sublist.get<std::string>("Model ID") == model_id)
            {
                side_pb_list = sublist;
            }
        }
    }
    else if (side_pb.getParameterList()->numParams() != 1)
    {
        const std::string msg
            = "\n\nERROR: Multiple equation sets are being solved on a single "
              "block but `Model ID` is not being specified in the "
              "corresponding boundary condition lists.\n\n";
        throw std::runtime_error(msg);
    }
    else
    {
        model_id
            = side_pb.getParameterList()->sublist("child0").get<std::string>(
                "Model ID");
        side_pb_list = side_pb.getParameterList()->sublist("child0");
    }
}

//---------------------------------------------------------------------------//
// Get integration basis for a variable with name `dof_name`.
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<panzer::BasisIRLayout>
BoundaryFluxBase<EvalType, NumSpaceDim>::getBasisIRLayout(
    const panzer::PhysicsBlock& /**side_pb**/, const std::string& dof_name) const
{
    const Teuchos::RCP<panzer::PureBasis> basis = _dof_basis_pair.at(dof_name);

    return panzer::basisIRLayout(basis, *_ir);
}

//---------------------------------------------------------------------------//
// Register degree of freedom and gradients for a variable with name `dof_name`
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerDOFsGradient(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const std::string& dof_name) const
{
    // Degree of freedom (DOF)
    const auto basis_layout = this->getBasisIRLayout(side_pb, dof_name);
    Teuchos::ParameterList dof_params;
    dof_params.set<std::string>("Name", dof_name);
    dof_params.set<Teuchos::RCP<panzer::BasisIRLayout>>("Basis", basis_layout);
    dof_params.set<Teuchos::RCP<panzer::IntegrationRule>>("IR", _ir);
    const auto dof_op
        = Teuchos::rcp(new panzer::DOF<EvalType, panzer::Traits>(dof_params));
    this->template registerEvaluator<EvalType>(fm, dof_op);

    // Gradient
    dof_params.set<std::string>("Gradient Name", "GRAD_" + dof_name);
    const auto gd_op = Teuchos::rcp(
        new panzer::DOFGradient<EvalType, panzer::Traits>(dof_params));
    this->template registerEvaluator<EvalType>(fm, gd_op);
}

//---------------------------------------------------------------------------//
// Register side nornal evaluator
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerSideNormals(
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb) const
{
    std::stringstream normal_params_name;
    normal_params_name << "Side Normal:" << side_pb.cellData().side();
    Teuchos::ParameterList normal_params(normal_params_name.str());
    normal_params.set<std::string>("Name", "Side Normal");
    normal_params.set<int>("Side ID", side_pb.cellData().side());
    normal_params.set<Teuchos::RCP<panzer::IntegrationRule>>("IR", _ir);
    normal_params.set<bool>("Normalize", true);
    const auto normal_op = Teuchos::rcp(
        new panzer::Normals<EvalType, panzer::Traits>(normal_params));
    this->template registerEvaluator<EvalType>(fm, normal_op);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerConvectionTypeFluxOperator(
    const std::pair<const std::string, std::string>& dof_eq_pair,
    std::unordered_map<std::string, std::vector<std::string>>& eq_res_map,
    const std::string& closure_name,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const double& multiplier) const
{
    // Local variables
    const auto& [eq_name, dof_name] = dof_eq_pair;
    const std::string normal_dot_name = "Normal Dot Flux " + eq_name;

    // Register dot product evaluator
    const std::string flux_name = "BOUNDARY_" + closure_name + "_FLUX_"
                                  + eq_name;
    const auto normal_dot_flux_op
        = panzer::buildEvaluator_DotProduct<EvalType, panzer::Traits>(
            normal_dot_name, *_ir, "Side Normal", flux_name);
    this->template registerEvaluator<EvalType>(fm, normal_dot_flux_op);

    // Convective flux integral.
    const auto basis_layout = this->getBasisIRLayout(side_pb, dof_name);
    const std::string residual_name = closure_name + "_BOUNDARY_RESIDUAL_"
                                      + eq_name;
    const auto convective_flux_op = Teuchos::rcp(
        new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
            panzer::EvaluatorStyle::EVALUATES,
            residual_name,
            normal_dot_name,
            *basis_layout,
            *_ir,
            multiplier));
    this->template registerEvaluator<EvalType>(fm, convective_flux_op);

    // Add residual name to `eq_res_map`
    eq_res_map[eq_name].push_back(residual_name);
}

//---------------------------------------------------------------------------//
// Register closure models used for the symmetric penalty method. Penalty
// parameters are specified in the `Data` sublist which is stored in
// `bc_params`.
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerPenaltyAndViscousGradientOperator(
    const std::pair<const std::string, std::string>& dof_eq_pair,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const Teuchos::ParameterList& bc_params) const
{
    // Local variables
    const auto& [eq_name, dof_name] = dof_eq_pair;
    const auto basis_layout = this->getBasisIRLayout(side_pb, dof_name);

    // Create viscous penalty parameter evaluator (just once).
    if (!_penalty_added)
    {
        // Check for boolean in boundary parameters
        bool use_updated_version = false;
        if (bc_params.isType<bool>("Use Updated Penalty Parameter"))
        {
            use_updated_version
                = bc_params.get<bool>("Use Updated Penalty Parameter");
        }

        // Register viscous penalty parameter evaluator and required metric
        // tensor
        if (use_updated_version)
        {
            const auto metric_tensor = Teuchos::rcp(
                new ClosureModel::MetricTensor<EvalType, panzer::Traits>(*_ir));
            this->template registerEvaluator<EvalType>(fm, metric_tensor);

            const auto viscous_penalty_op = Teuchos::rcp(
                new ViscousPenaltyParameterMetricTensor<EvalType, panzer::Traits>(
                    *_ir));
            this->template registerEvaluator<EvalType>(fm, viscous_penalty_op);
        }
        else
        {
            const auto viscous_penalty_op = Teuchos::rcp(
                new ViscousPenaltyParameter<EvalType, panzer::Traits>(
                    *_ir, *(basis_layout->getBasis())));
            this->template registerEvaluator<EvalType>(fm, viscous_penalty_op);
        }
        _penalty_added = true;
    }

    // Create boundary gradients.
    {
        // Check for user-specified penalty parameter
        double penalty = 10.0;

        // For temperature, default penalty is set to 1000.0
        if (dof_name == "temperature")
            penalty = 1000.0;

        if (bc_params.isSublist("Penalty Parameters"))
        {
            const auto penalty_list = bc_params.sublist("Penalty Parameters");
            if (penalty_list.isType<double>(dof_name))
            {
                penalty = penalty_list.get<double>(dof_name);
            }
        }

        const auto viscous_gradient_op
            = Teuchos::rcp(new ViscousGradient<EvalType, panzer::Traits>(
                *_ir, dof_name, penalty));
        this->template registerEvaluator<EvalType>(fm, viscous_gradient_op);
    }
}

//---------------------------------------------------------------------------//
// Register Laplace-type operators
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerViscousTypeFluxOperator(
    const std::pair<const std::string, std::string>& dof_eq_pair,
    std::unordered_map<std::string, std::vector<std::string>>& eq_res_map,
    const std::string& closure_name,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb,
    const double& multiplier) const
{
    // Local variables
    const auto& [eq_name, dof_name] = dof_eq_pair;
    const auto basis_layout = this->getBasisIRLayout(side_pb, dof_name);

    // Symmetric interior penalty method residual 1.
    {
        const std::string normal_dot_viscous_name
            = "Normal Dot " + closure_name + " Flux " + eq_name;
        const std::string flux_name = "BOUNDARY_" + closure_name + "_FLUX_"
                                      + eq_name;
        auto normal_dot_viscous_flux_op
            = panzer::buildEvaluator_DotProduct<EvalType, panzer::Traits>(
                normal_dot_viscous_name, *_ir, "Side Normal", flux_name);
        this->template registerEvaluator<EvalType>(fm,
                                                   normal_dot_viscous_flux_op);

        const std::string bnd_resid = closure_name + "_BOUNDARY_RESIDUAL_"
                                      + eq_name;
        auto bnd_op = Teuchos::rcp(
            new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                bnd_resid,
                normal_dot_viscous_name,
                *basis_layout,
                *_ir,
                -multiplier));
        this->template registerEvaluator<EvalType>(fm, bnd_op);
        eq_res_map[eq_name].push_back(bnd_resid);
    }

    // Symmetric interior penalty method residual 2.
    {
        const std::string penalty_bnd_resid = "PENALTY_BOUNDARY_RESIDUAL_"
                                              + eq_name;
        auto bnd_penalty_op = Teuchos::rcp(
            new Integrator::BoundaryGradBasisDotVector<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                penalty_bnd_resid,
                "PENALTY_BOUNDARY_" + closure_name + "_FLUX_" + eq_name,
                *basis_layout,
                *_ir,
                -multiplier));
        this->template registerEvaluator<EvalType>(fm, bnd_penalty_op);
        eq_res_map[eq_name].push_back(penalty_bnd_resid);
    }

    // Symmetric interior penalty method residual 3.
    {
        const std::string normal_dot_scaled_penalty_viscous_name
            = "Normal Dot Scaled Penalty " + closure_name + " Flux " + eq_name;
        const std::string scaled_penalty_flux_name
            = "SYMMETRY_BOUNDARY_" + closure_name + "_FLUX_" + eq_name;
        auto normal_dot_scaled_penalty_viscous_flux_op
            = panzer::buildEvaluator_DotProduct<EvalType, panzer::Traits>(
                normal_dot_scaled_penalty_viscous_name,
                *_ir,
                "Side Normal",
                scaled_penalty_flux_name);
        this->template registerEvaluator<EvalType>(
            fm, normal_dot_scaled_penalty_viscous_flux_op);

        const std::string scaled_penalty_bnd_resid
            = "SYMMETRY_BOUNDARY_RESIDUAL_" + eq_name;
        auto scaled_bnd_penalty_op = Teuchos::rcp(
            new panzer::Integrator_BasisTimesScalar<EvalType, panzer::Traits>(
                panzer::EvaluatorStyle::EVALUATES,
                scaled_penalty_bnd_resid,
                normal_dot_scaled_penalty_viscous_name,
                *basis_layout,
                *_ir,
                multiplier));
        this->template registerEvaluator<EvalType>(fm, scaled_bnd_penalty_op);
        eq_res_map[eq_name].push_back(scaled_penalty_bnd_resid);
    }
}

//---------------------------------------------------------------------------//
// Register residuals collected in `eq_res_map` for each equation.
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerResidual(
    std::pair<const std::string, std::string> dof_eq_pair,
    std::unordered_map<std::string, std::vector<std::string>>& eq_res_map,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& side_pb) const
{
    // Local variables
    const auto& [eq_name, dof_name] = dof_eq_pair;
    const auto basis_layout = this->getBasisIRLayout(side_pb, dof_name);

    // Initialize residual vector
    auto residuals = Teuchos::rcp(new std::vector<std::string>);
    for (auto& res : eq_res_map[eq_name])
        residuals->push_back(res);

    // Register residuals
    Teuchos::ParameterList sum_params;
    sum_params.set("Sum Name", "BOUNDARY_RESIDUAL_" + eq_name);
    sum_params.set("Values Names", residuals);
    sum_params.set("Data Layout", basis_layout->getBasis()->functional);
    auto sum_op = Teuchos::rcp(
        new panzer::SumStatic<EvalType, panzer::Traits, panzer::Cell, panzer::BASIS>(
            sum_params));
    this->template registerEvaluator<EvalType>(fm, sum_op);
}

//---------------------------------------------------------------------------//
// Register and scatter residual for each equation
template<class EvalType, int NumSpaceDim>
void BoundaryFluxBase<EvalType, NumSpaceDim>::registerScatterOperator(
    std::pair<const std::string, std::string> dof_eq_pair,
    PHX::FieldManager<panzer::Traits>& fm,
    const panzer::PhysicsBlock& /**side_pb**/,
    const panzer::LinearObjFactory<panzer::Traits>& lof) const
{
    const auto& [eq_name, dof_name] = dof_eq_pair;
    const Teuchos::RCP<const panzer::PureBasis> basis
        = _dof_basis_pair.at(dof_name);

    const std::string residual_name = "BOUNDARY_RESIDUAL_" + eq_name;
    Teuchos::ParameterList p("Scatter: " + residual_name + " to " + eq_name);

    const std::string scatter_field_name
        = "Dummy Scatter: " + this->m_bc.identifier() + residual_name;
    p.set("Scatter Name", scatter_field_name);
    p.set("Basis", basis);

    auto residual_names = Teuchos::rcp(new std::vector<std::string>);
    residual_names->push_back(residual_name);
    p.set("Dependent Names", residual_names);

    auto names_map = Teuchos::rcp(new std::map<std::string, std::string>);
    names_map->insert(
        std::pair<std::string, std::string>(residual_name, dof_name));
    p.set("Dependent Map", names_map);

    auto scatter_op = lof.buildScatter<EvalType>(p);

    this->template registerEvaluator<EvalType>(fm, scatter_op);

    // Require variables
    auto dummy_layout = Teuchos::rcp(new PHX::MDALayout<panzer::Dummy>(0));
    const PHX::Tag<typename EvalType::ScalarT> tag(scatter_field_name,
                                                   dummy_layout);
    fm.template requireField<EvalType>(tag);
}
//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYCONDITION_BOUNDARYFLUXBASE_IMPL_HPP
