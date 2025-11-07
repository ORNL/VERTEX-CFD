#include "VertexCFD_InitialConditionManager.hpp"

namespace VertexCFD
{
//---------------------------------------------------------------------------//
InitialConditionManager::InitialConditionManager(
    const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
    const Teuchos::RCP<MeshManager>& mesh_manager)
    : _parameter_db(parameter_db)
    , _mesh_manager(mesh_manager)
    , _do_restart(false)
    , _t_init(0.0)
{
    // Create restart reader if needed and get the starting time.
    auto read_restart_params = _parameter_db->readRestartParameters();
    if (Teuchos::nonnull(read_restart_params))
    {
        if (read_restart_params->isType<bool>("Read Restart"))
        {
            _do_restart = read_restart_params->get<bool>("Read Restart");
        }
        if (_do_restart)
        {
            auto comm = mesh_manager->comm();
            _restart_reader = Teuchos::rcp(new VertexCFD::Mesh::RestartReader(
                comm, *read_restart_params));
            _t_init = _restart_reader->initialStateTime();
        }
    }
}

//---------------------------------------------------------------------------//
double InitialConditionManager::initialTime() const
{
    return _t_init;
}

//---------------------------------------------------------------------------//
template<int NumSpaceDim>
void InitialConditionManager::applyInitialConditions(
    const std::integral_constant<int, NumSpaceDim>&,
    const PhysicsManager& physics_manager,
    Teuchos::RCP<Thyra::VectorBase<double>>& x,
    Teuchos::RCP<Thyra::VectorBase<double>>& x_dot) const
{
    // Initialize 'num_space_dim' with template value
    constexpr int num_space_dim = NumSpaceDim;

    // Create vectors if needed.
    if (Teuchos::is_null(x))
    {
        x = Thyra::createMember(
            physics_manager.modelEvaluator()->get_x_space());
    }
    if (Teuchos::is_null(x_dot))
    {
        x_dot = Thyra::createMember(
            physics_manager.modelEvaluator()->get_x_space());
    }

    // Set initial conditions from restart.
    if (_do_restart)
    {
        auto mesh = _mesh_manager->mesh();
        auto dof_manager = physics_manager.dofManager();
        _restart_reader->readSolution(mesh, dof_manager, x, x_dot);
    }

    // Set initial conditions from input.
    else
    {
        auto workset_container = physics_manager.worksetContainer();
        auto linear_object_factory = physics_manager.linearObjectFactory();
        auto physics_blocks = physics_manager.physicsBlocks();
        auto user_params = _parameter_db->userParameters();
        auto ic_params = _parameter_db->initialConditionParameters();
        const bool write_graph = user_params->get<bool>("Output Graph");
        const VertexCFD::InitialCondition::FactoryTemplateBuilder<num_space_dim>
            ic_builder(_mesh_manager->mesh());
        panzer::ClosureModelFactory_TemplateManager<panzer::Traits> ic_factory;
        ic_factory.buildObjects(ic_builder);
        std::map<std::string, Teuchos::RCP<PHX::FieldManager<panzer::Traits>>>
            phx_ic_field_managers;
        panzer::setupInitialConditionFieldManagers(*workset_container,
                                                   physics_blocks,
                                                   ic_factory,
                                                   *ic_params,
                                                   *linear_object_factory,
                                                   *user_params,
                                                   write_graph,
                                                   "",
                                                   phx_ic_field_managers);
        const Teuchos::RCP<panzer::LinearObjContainer> linear_object_container
            = linear_object_factory->buildLinearObjContainer();
        const Teuchos::RCP<panzer::ThyraObjContainer<double>> thyra_loc
            = Teuchos::rcp_dynamic_cast<panzer::ThyraObjContainer<double>>(
                linear_object_container);
        thyra_loc->set_x_th(x);
        panzer::evaluateInitialCondition(*workset_container,
                                         phx_ic_field_managers,
                                         linear_object_container,
                                         *linear_object_factory,
                                         _t_init);
        Thyra::assign(x_dot.ptr(), 0.0);
    }
}

//---------------------------------------------------------------------------//
// Explicit instantiation of 'applyInitialConditions'
template void InitialConditionManager::applyInitialConditions(
    const std::integral_constant<int, 2>&,
    const PhysicsManager& physics_manager,
    Teuchos::RCP<Thyra::VectorBase<double>>& x,
    Teuchos::RCP<Thyra::VectorBase<double>>& x_dot) const;

template void InitialConditionManager::applyInitialConditions(
    const std::integral_constant<int, 3>&,
    const PhysicsManager& physics_manager,
    Teuchos::RCP<Thyra::VectorBase<double>>& x,
    Teuchos::RCP<Thyra::VectorBase<double>>& x_dot) const;

//---------------------------------------------------------------------------//

} // end namespace VertexCFD
