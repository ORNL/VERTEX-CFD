#ifndef VERTEXCFD_COMPUTE_VOLUME_IMPL_HPP
#define VERTEXCFD_COMPUTE_VOLUME_IMPL_HPP

#include <Panzer_ResponseEvaluatorFactory_Functional.hpp>

#include <Teuchos_DefaultMpiComm.hpp>

#include <string>
#include <vector>

namespace VertexCFD
{
namespace ComputeVolume
{
//---------------------------------------------------------------------------//
template<class Scalar>
Volume<Scalar>::Volume(
    const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
    const Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits>>& lof,
    const Teuchos::RCP<panzer::ResponseLibrary<panzer::Traits>>& response_library,
    const std::vector<Teuchos::RCP<panzer::PhysicsBlock>>& physics_blocks,
    const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& cm_factory,
    const Teuchos::ParameterList& closure_params,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::WorksetContainer>& workset_container,
    const std::vector<panzer::BC>& bcs,
    const Teuchos::RCP<panzer::BCStrategyFactory>& bc_factory,
    const Teuchos::RCP<panzer::EquationSetFactory>& eq_set_factory,
    const int integration_order)
    : _lof(lof)
    , _response_library(response_library)
    , _physics_blocks(physics_blocks)
    , _cm_factory(cm_factory)
    , _closure_params(closure_params)
    , _user_params(user_params)
    , _workset_container(workset_container)
    , _bcs(bcs)
    , _bc_factory(bc_factory)
    , _eq_set_factory(eq_set_factory)
{
    // Add response library
    std::vector<std::string> element_blocks;
    mesh->getElementBlockNames(element_blocks);

    panzer::FunctionalResponse_Builder<int, int> response_builder_vol;
    response_builder_vol.comm = Teuchos::getRawMpiComm(*(mesh->getComm()));
    response_builder_vol.cubatureDegree = integration_order;
    response_builder_vol.requiresCellIntegral = true;

    response_builder_vol.quadPointField = "volume";

    _response_library->addResponse(
        "compute volume", element_blocks, response_builder_vol);

    // Finalize construction of response library
    _response_library->buildResponseEvaluators(
        _physics_blocks, _cm_factory, _closure_params, _user_params);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void Volume<Scalar>::ComputeVol()
{
    // Assemble linear system
    panzer::AssemblyEngineInArgs in_args;
    in_args.container_ = _lof->buildLinearObjContainer();
    in_args.ghostedContainer_ = _lof->buildGhostedLinearObjContainer();
    in_args.evaluate_transient_terms = false;
    in_args.time = 0.0;
    in_args.step_size = 0.0;

    _lof->initializeGhostedContainer(panzer::LinearObjContainer::X
                                         | panzer::LinearObjContainer::F
                                         | panzer::LinearObjContainer::Mat,
                                     *(in_args.ghostedContainer_));

    // Set up resp, resp_func, resp_vec for volume
    auto resp_vol = _response_library->getResponse<panzer::Traits::Residual>(
        "compute volume");
    auto resp_func_vol = Teuchos::rcp_dynamic_cast<
        panzer::Response_Functional<panzer::Traits::Residual>>(resp_vol);
    auto resp_vec_vol = Thyra::createMember(resp_func_vol->getVectorSpace());
    resp_func_vol->setVector(resp_vec_vol);

    _response_library->addResponsesToInArgs<panzer::Traits::Residual>(in_args);
    _response_library->evaluate<panzer::Traits::Residual>(in_args);

    _volume = resp_func_vol->value;
}

//---------------------------------------------------------------------------//
template<class Scalar>
double Volume<Scalar>::volume() const
{
    return _volume;
}
//---------------------------------------------------------------------------//

} // end namespace ComputeVolume
} // end namespace VertexCFD

#endif // end VERTEXCFD_COMPUTE_VOLUME_IMPL_HPP
