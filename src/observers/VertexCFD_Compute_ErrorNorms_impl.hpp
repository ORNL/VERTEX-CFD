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
    // Check which equation set is used  by scanning the closure model list.
    // NOTE: we may want to update ComputeErrorNorms to require that
    // equations to be included in the error norm output be explicitly
    // listed in the "User Data -> Compute Error Norms" input.
    bool build_ns_error_norms = true;
    bool build_rad_error_norms = false;
    bool build_induction_error_norms = false;
    bool build_lsvof_error_norms = false;
    for (auto it = closure_params.begin(); it != closure_params.end(); ++it)
    {
        const std::string name = it->first;
        if (closure_params.isSublist(name))
        {
            const auto p = closure_params.sublist(name);
            if (p.isType<std::string>("Closure Factory Type"))
            {
                if (p.get<std::string>("Closure Factory Type")
                    == "Full Induction MHD")
                {
                    build_induction_error_norms = true;
                }
                else if (p.get<std::string>("Closure Factory Type") == "LSVOF")
                {
                    build_lsvof_error_norms = true;
                }
                else if (p.get<std::string>("Closure Factory Type") == "RAD")
                {
                    build_rad_error_norms = true;
                }
            }
        }
        if (build_induction_error_norms || build_lsvof_error_norms
            || build_rad_error_norms)
            break;
    }

    // If solving RAD equation, disable NS equations
    if (build_rad_error_norms)
        build_ns_error_norms = false;

    // If using LSVOF model, check if NS equation is disabled
    if (build_lsvof_error_norms)
    {
        for (auto it = closure_params.begin(); it != closure_params.end(); ++it)
        {
            const std::string name = it->first;
            if (closure_params.isSublist(name))
            {
                const auto p = closure_params.sublist(name);
                if (p.isSublist("LSVOF_Properties"))
                {
                    const Teuchos::ParameterList& lsvof_params
                        = p.sublist("LSVOF_Properties");

                    if (lsvof_params.isType<bool>("Build LSVOF Navier-Stokes "
                                                  "Equations"))
                    {
                        build_ns_error_norms = lsvof_params.get<bool>(
                            "Build LSVOF Navier-Stokes Equations");
                    }
                }
            }
        }
    }

    // Initialize names of equation based on the equation set name
    std::vector<std::string> eq_name;
    _num_space_dim = mesh->getDimension();

    // If solving RAD equation, Add species
    if (build_rad_error_norms)
    {
        for (auto it = closure_params.begin(); it != closure_params.end(); ++it)
        {
            const std::string name = it->first;
            if (closure_params.isSublist(name))
            {
                const auto p = closure_params.sublist(name);
                const auto rad_params = p.sublist("Model Parameters");
                const int num_species
                    = rad_params.isType<int>("Number of Species")
                          ? rad_params.get<int>("Number of Species")
                          : 1;
                for (int i = 0; i < num_species; ++i)
                {
                    eq_name.push_back("reaction_advection_diffusion_equation_"
                                      + std::to_string(i));
                }
            }
        }
    }

    // NOTE: ordering of error norms in 'eq_name' must be preserved for
    // regression tests to pass. New error terms should be added to the end of
    // the list after the existing terms.

    // Add Continuity first
    if (build_ns_error_norms)
    {
        eq_name.push_back("continuity");
    }

    // Temperature equation
    if (_user_params.isType<bool>("Build Temperature Equation"))
    {
        if (_user_params.get<bool>("Build Temperature Equation"))
            eq_name.push_back("energy");
    }

    // Induction less equation
    if (_user_params.isType<bool>("Build Inductionless MHD Equation"))
    {
        if (_user_params.get<bool>("Build Inductionless MHD Equation"))
            eq_name.push_back("electric_potential_equation");
    }

    // Full induction equation
    if (build_induction_error_norms)
    {
        for (int i = 0; i < _num_space_dim; ++i)
        {
            eq_name.push_back("induction_" + std::to_string(i));
        }
    }

    // Momentum equation
    if (build_ns_error_norms)
    {
        // Momentum equation
        for (int i = 0; i < _num_space_dim; ++i)
            eq_name.push_back("momentum_" + std::to_string(i));
    }

    // LSVOF scalar equations
    if (build_lsvof_error_norms)
    {
        for (auto it = closure_params.begin(); it != closure_params.end(); ++it)
        {
            const std::string name = it->first;
            if (closure_params.isSublist(name))
            {
                const auto p = closure_params.sublist(name);
                if (p.isSublist("LSVOF_Properties"))
                {
                    const Teuchos::ParameterList& lsvof_params
                        = p.sublist("LSVOF_Properties");

                    const std::string lsvof_model
                        = lsvof_params.get<std::string>("LSVOF Model");

                    const int num_phases
                        = lsvof_params.get<int>("Number of Phases");

                    if (lsvof_model == "VOF")
                    {
                        for (int phase = 1; phase < num_phases; ++phase)
                        {
                            const Teuchos::ParameterList& phase_list
                                = lsvof_params.sublist(
                                    "Phase " + std::to_string(phase));

                            const std::string phase_name
                                = phase_list.get<std::string>("Phase Name");

                            eq_name.push_back("alpha_" + phase_name);
                        }
                    }
                }
            }
        }
    }

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
