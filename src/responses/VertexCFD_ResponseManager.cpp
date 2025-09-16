// Due to a conflict with FAD types and Kokkos views, this file needs to be
// included before other VertexCFD includes
#include "utils/VertexCFD_Utils_KokkosFadFixup.hpp"

#include "VertexCFD_ResponseManager.hpp"

#include <PanzerCore_config.hpp>
#include <Panzer_ParameterLibraryUtilities.hpp>
#include <Panzer_ResponseEvaluatorFactory_ExtremeValue.hpp>
#include <Panzer_ResponseEvaluatorFactory_Functional.hpp>
#include <Panzer_ResponseEvaluatorFactory_Probe.hpp>

#include <Thyra_VectorSpaceBase.hpp>
#include <Thyra_VectorStdOps.hpp>

#include <Teuchos_DefaultMpiComm.hpp>

#include <algorithm>

namespace VertexCFD
{
namespace Response
{
//---------------------------------------------------------------------------//
ResponseManager::ResponseManager(Teuchos::RCP<PhysicsManager> physics_manager)
    : _num_responses(0)
    , _physics_manager(physics_manager)
{
    std::vector<std::string> element_blocks;
    _physics_manager->meshManager()->mesh()->getElementBlockNames(
        element_blocks);

    _default_workset_descriptors.reserve(element_blocks.size());
    for (const auto& block : element_blocks)
        _default_workset_descriptors.emplace_back(block);
}

//---------------------------------------------------------------------------//
int ResponseManager::numResponses() const
{
    return _num_responses;
}

//---------------------------------------------------------------------------//
void ResponseManager::addFunctionalResponse(const std::string& name,
                                            const std::string& field_name)
{
    addFunctionalResponse(name, field_name, _default_workset_descriptors);
}

//---------------------------------------------------------------------------//
void ResponseManager::addFunctionalResponse(
    const std::string& name,
    const std::string& field_name,
    const std::vector<panzer::WorksetDescriptor>& workset_descriptors)
{
    // Setup the response builder.
    auto builder = Teuchos::rcp(
        new panzer::FunctionalResponse_Builder<panzer::LocalOrdinal,
                                               panzer::GlobalOrdinal>);
    builder->comm
        = Teuchos::getRawMpiComm(*_physics_manager->meshManager()->comm());
    builder->cubatureDegree = _physics_manager->integrationOrder();
    builder->requiresCellIntegral = true;
    builder->quadPointField = field_name;
    builder->applyDirichletToDerivative = false;

    addResponseFromBuilder(name, workset_descriptors, builder);
}

//---------------------------------------------------------------------------//
void ResponseManager::addMinValueResponse(const std::string& name,
                                          const std::string& field_name)
{
    addMinValueResponse(name, field_name, _default_workset_descriptors);
}

//---------------------------------------------------------------------------//
void ResponseManager::addMinValueResponse(
    const std::string& name,
    const std::string& field_name,
    const std::vector<panzer::WorksetDescriptor>& workset_descriptors)
{
    constexpr int use_max = false;
    addExtremeValueResponse(use_max, name, field_name, workset_descriptors);
}

//---------------------------------------------------------------------------//
void ResponseManager::addMaxValueResponse(const std::string& name,
                                          const std::string& field_name)
{
    addMaxValueResponse(name, field_name, _default_workset_descriptors);
}

//---------------------------------------------------------------------------//
void ResponseManager::addMaxValueResponse(
    const std::string& name,
    const std::string& field_name,
    const std::vector<panzer::WorksetDescriptor>& workset_descriptors)
{
    constexpr int use_max = true;
    addExtremeValueResponse(use_max, name, field_name, workset_descriptors);
}

//---------------------------------------------------------------------------//
void ResponseManager::addExtremeValueResponse(
    const bool use_max,
    const std::string& name,
    const std::string& field_name,
    const std::vector<panzer::WorksetDescriptor>& workset_descriptors)
{
    // Setup the response builder.
    auto builder = Teuchos::rcp(
        new panzer::ExtremeValueResponse_Builder<panzer::LocalOrdinal,
                                                 panzer::GlobalOrdinal>);
    builder->comm
        = Teuchos::getRawMpiComm(*_physics_manager->meshManager()->comm());
    builder->cubatureDegree = _physics_manager->integrationOrder();
    builder->requiresCellExtreme = true;
    builder->useMax = use_max;
    builder->quadPointField = field_name;
    builder->applyDirichletToDerivative = false;
    builder->prefix = use_max ? "Max " : "Min ";

    addResponseFromBuilder(name, workset_descriptors, builder);
}

//---------------------------------------------------------------------------//
void ResponseManager::addProbeResponse(const std::string& name,
                                       const std::string& field_name,
                                       const Teuchos::Array<double>& point)
{
    addProbeResponse(name, field_name, point, _default_workset_descriptors);
}

//---------------------------------------------------------------------------//
void ResponseManager::addProbeResponse(
    const std::string& name,
    const std::string& field_name,
    const Teuchos::Array<double>& point,
    const std::vector<panzer::WorksetDescriptor>& workset_descriptors)
{
    // Setup the response builder.
    auto builder = Teuchos::rcp(
        new panzer::ProbeResponse_Builder<panzer::LocalOrdinal,
                                          panzer::GlobalOrdinal>);
    builder->comm
        = Teuchos::getRawMpiComm(*_physics_manager->meshManager()->comm());
    builder->cubatureDegree = _physics_manager->integrationOrder();
    builder->fieldName = field_name;
    builder->fieldComponent = 0;
    builder->point = point;
    builder->applyDirichletToDerivative = false;

    addResponseFromBuilder(name, workset_descriptors, builder);
}

//---------------------------------------------------------------------------//
template<class Builder>
void ResponseManager::addResponseFromBuilder(
    const std::string& name,
    const std::vector<panzer::WorksetDescriptor>& workset_descriptors,
    const Builder& builder)
{
    auto model_evaluator = _physics_manager->modelEvaluator();

    // If no workset descriptors were provided, use the default
    // (all element blocks).
    const auto& workset_desc = workset_descriptors.empty()
                                   ? _default_workset_descriptors
                                   : workset_descriptors;

    const int response_index
        = model_evaluator->addFlexibleResponse(name, workset_desc, builder);

    _resp_vectors.emplace_back(
        Thyra::createMember(model_evaluator->get_g_space(response_index)));

    _index_map.emplace_back(response_index);
    _name_map.emplace(name, _num_responses);
    _is_active.emplace_back(true);

    // Add a scalar parameter to store the result of the response. Default
    // value is zero.
    panzer::registerScalarParameter(
        name, *(_physics_manager->globalData()->pl), 0.0);

    ++_num_responses;
}

//---------------------------------------------------------------------------//
void ResponseManager::activateResponse(const int index)
{
    _is_active.at(index) = true;
}

//---------------------------------------------------------------------------//
void ResponseManager::activateResponse(const std::string& name)
{
    const int index = _name_map.at(name);
    activateResponse(index);
}

//---------------------------------------------------------------------------//
void ResponseManager::deactivateAll()
{
    std::fill(_is_active.begin(), _is_active.end(), false);
}

//---------------------------------------------------------------------------//
void ResponseManager::evaluateResponses(
    const Teuchos::RCP<Thyra::VectorBase<double>>& x,
    const Teuchos::RCP<Thyra::VectorBase<double>>& x_dot,
    const double t,
    const double dt)
{
    const int num_active
        = std::count(_is_active.begin(), _is_active.end(), true);

    // Just return if there are no active responses.
    if (num_active == 0)
        return;

    auto model_evaluator = _physics_manager->modelEvaluator();

    auto in_args = model_evaluator->createInArgs();
    auto out_args = model_evaluator->createOutArgs();

    in_args.set_t(t);
    in_args.set_step_size(dt);
    in_args.set_x(x);
    in_args.set_x_dot(x_dot);

    // Set output vector for each active response.
    for (int i = 0; i < _num_responses; ++i)
    {
        if (_is_active[i])
        {
            out_args.set_g(_index_map[i], _resp_vectors[i]);
        }
    }

    // Evaluate the response.
    model_evaluator->evalModel(in_args, out_args);

    // Extract the value and insert it into the parameter library.
    for (int i = 0; i < _num_responses; ++i)
    {
        panzer::registerScalarParameter(
            name(i), *(_physics_manager->globalData()->pl), value(i));
    }
}

//---------------------------------------------------------------------------//
int ResponseManager::globalIndex(const int index) const
{
    return _index_map.at(index);
}

//---------------------------------------------------------------------------//
int ResponseManager::globalIndex(const std::string& name) const
{
    const int index = _name_map.at(name);
    return globalIndex(index);
}

//---------------------------------------------------------------------------//
const std::string& ResponseManager::name(const int index) const
{
    return _physics_manager->modelEvaluator()->get_g_name(globalIndex(index));
}

//---------------------------------------------------------------------------//
double ResponseManager::value(const int index) const
{
    return Thyra::get_ele(*_resp_vectors.at(index), 0);
}

//---------------------------------------------------------------------------//
double ResponseManager::value(const std::string& name) const
{
    const int index = _name_map.at(name);
    return value(index);
}

//---------------------------------------------------------------------------//

} // namespace Response
} // namespace VertexCFD
