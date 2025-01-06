#ifndef VERTEXCFD_COMPUTE_ERRORNORMS_IMPL_HPP
#define VERTEXCFD_COMPUTE_ERRORNORMS_IMPL_HPP

#include <Panzer_ResponseEvaluatorFactory_Functional.hpp>

#include <Teuchos_DefaultMpiComm.hpp>

#include <string>
#include <vector>

namespace VertexCFD
{
namespace ComputeErrorNorms
{
//---------------------------------------------------------------------------//
template<class Scalar>
ErrorNorms<Scalar>::ErrorNorms(
    const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
    const Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>>& lof,
    const Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>>& response_library,
    const std::vector<Teuchos::RCP<panzer::PhysicsBlock>>& physics_blocks,
    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& cm_factory,
    const Teuchos::ParameterList& closure_params,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::EquationSetFactory>& eq_set_factory,
    const double volume,
    const int integration_order)
    : _lof(lof)
    , _response_library(response_library)
    , _physics_blocks(physics_blocks)
    , _cm_factory(cm_factory)
    , _closure_params(closure_params)
    , _user_params(user_params)
    , _eq_set_factory(eq_set_factory)
    , _volume(volume)
{
    // Initialize names of equation based on the equation set name
    std::vector<std::string> eq_name;
    _num_mom_eq = mesh->getDimension();
    // Continuity equation
    eq_name.push_back("continuity");
    // Temperature equation
    if (_user_params.isType<bool>("Build Temperature Equation"))
    {
        if (_user_params.get<bool>("Build Temperature Equation"))
            eq_name.push_back("energy");
    }
    // Full induction equation
    if (_user_params.isSublist("Full Induction MHD Properties"))
    {
        for (int i = 0; i < _num_mom_eq; ++i)
        {
            eq_name.push_back("induction_" + std::to_string(i));
        }
    }
    // Momentum equation
    for (int i = 0; i < _num_mom_eq; ++i)
        eq_name.push_back("momentum_" + std::to_string(i));

    // Add response library
    std::vector<std::string> element_blocks;
    mesh->getElementBlockNames(element_blocks);

    panzer::FunctionalResponse_Builder<int, int> response_builder;
    response_builder.comm = Teuchos::getRawMpiComm(*(mesh->getComm()));
    response_builder.cubatureDegree = integration_order;
    response_builder.requiresCellIntegral = true;

    // L1 error norm
    auto add_dof_L1 = [&](const std::string& dof) {
        response_builder.quadPointField = "L1_Error_" + dof;
        _response_library->addResponse(
            dof + "_L1_error_norms", element_blocks, response_builder);

        _L1_error_norms.emplace_back(dof);
    };

    // L2 error norm
    auto add_dof_L2 = [&](const std::string& dof) {
        response_builder.quadPointField = "L2_Error_" + dof;
        _response_library->addResponse(
            dof + "_L2_error_norms", element_blocks, response_builder);

        _L2_error_norms.emplace_back(dof);
    };

    // Add L1 and L2 error norms
    for (auto& element : eq_name)
    {
        add_dof_L1(element);
        add_dof_L2(element);
    }

    // Finalize construction of response library
    _response_library->buildResponseEvaluators(
        _physics_blocks, _cm_factory, _closure_params, _user_params);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ErrorNorms<Scalar>::ComputeNorms(
    const Teuchos::RCP<Tempus::SolutionState<Scalar>>& working_state)
{
    // Get the working X
    auto x = working_state->getX();

    // Assemble linear system
    panzer::AssemblyEngineInArgs in_args;
    in_args.container_ = _lof->buildLinearObjContainer();
    in_args.ghostedContainer_ = _lof->buildGhostedLinearObjContainer();
    in_args.evaluate_transient_terms = false;
    in_args.time = working_state->getTime();
    in_args.step_size = working_state->getTimeStep();

    _lof->initializeGhostedContainer(panzer::LinearObjContainer::X,
                                     *(in_args.ghostedContainer_));

    auto thyra_container
        = Teuchos::rcp_dynamic_cast<panzer::ThyraObjContainer<double>>(
            in_args.container_);
    thyra_container->set_x_th(
        Teuchos::rcp_const_cast<Thyra::VectorBase<double>>(x));

    // Set up resp, resp_func, resp_vec for L1_error_norms
    for (auto& dof : _L1_error_norms)
    {
        auto resp = _response_library->getResponse<panzer::Traits::Residual>(
            dof.name + "_L1_error_norms");
        auto resp_func = Teuchos::rcp_dynamic_cast<
            panzer::Response_Functional<panzer::Traits::Residual>>(resp);
        auto resp_vec = Thyra::createMember(resp_func->getVectorSpace());
        resp_func->setVector(resp_vec);
    }

    // Set up resp, resp_func, resp_vec for L2_error_norms
    for (auto& dof : _L2_error_norms)
    {
        auto resp = _response_library->getResponse<panzer::Traits::Residual>(
            dof.name + "_L2_error_norms");
        auto resp_func = Teuchos::rcp_dynamic_cast<
            panzer::Response_Functional<panzer::Traits::Residual>>(resp);
        auto resp_vec = Thyra::createMember(resp_func->getVectorSpace());
        resp_func->setVector(resp_vec);
    }

    _response_library->addResponsesToInArgs<panzer::Traits::Residual>(in_args);
    _response_library->evaluate<panzer::Traits::Residual>(in_args);

    // Compute L1 error norm
    for (auto& dof : _L1_error_norms)
    {
        auto resp = _response_library->getResponse<panzer::Traits::Residual>(
            dof.name + "_L1_error_norms");
        auto resp_func = Teuchos::rcp_dynamic_cast<
            panzer::Response_Functional<panzer::Traits::Residual>>(resp);
        dof.error_norm = (resp_func->value) / _volume;
    }

    // Compute L2 error norm
    for (auto& dof : _L2_error_norms)
    {
        auto resp = _response_library->getResponse<panzer::Traits::Residual>(
            dof.name + "_L2_error_norms");
        auto resp_func = Teuchos::rcp_dynamic_cast<
            panzer::Response_Functional<panzer::Traits::Residual>>(resp);
        dof.error_norm = std::sqrt(resp_func->value) / _volume;
    }
}

//---------------------------------------------------------------------------//

} // end namespace ComputeErrorNorms
} // end namespace VertexCFD

#endif // end VERTEXCFD_COMPUTE_ERRORNORMS_IMPL_HPP
