#ifndef VERTEXCFD_PHYSICSMANAGER_HPP
#define VERTEXCFD_PHYSICSMANAGER_HPP

#include "VertexCFD_MeshManager.hpp"

#include "parameters/VertexCFD_ParameterDatabase.hpp"

#include <Panzer_BCStrategy_Factory.hpp>
#include <Panzer_BCStrategy_Factory_Composite.hpp>
#include <Panzer_ClosureModel_Factory.hpp>
#include <Panzer_EquationSet_Factory.hpp>
#include <Panzer_EquationSet_Factory_Composite.hpp>
#include <Panzer_GlobalData.hpp>
#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_LinearObjFactory.hpp>
#include <Panzer_ModelEvaluator.hpp>
#include <Panzer_PhysicsBlock.hpp>
#include <Panzer_WorksetContainer.hpp>

#include <Teuchos_RCP.hpp>

#include <type_traits>
#include <unordered_map>
#include <vector>

namespace VertexCFD
{
//---------------------------------------------------------------------------//
class PhysicsManager
{
  public:
    template<int NumSpaceDim>
    PhysicsManager(const std::integral_constant<int, NumSpaceDim>&,
                   const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
                   const Teuchos::RCP<MeshManager>& mesh_manager,
                   const double initial_time = 0.0);

    // Add a scalar parameter and an initial value for the parameter. Return
    // the parameter index.
    int addScalarParameter(const std::string& name, const double value);

    // Get the index of a parameter with the given name.
    int getParameterIndex(const std::string& name) const;

    // Setup the model after all physics responses and parameters have been
    // added.
    void setupModel();

    // Data accessors.
    Teuchos::RCP<MeshManager> meshManager() const;
    Teuchos::RCP<panzer::GlobalData> globalData() const;
    Teuchos::RCP<panzer::EquationSetFactory> equationSetFactory() const;
    const std::vector<Teuchos::RCP<panzer::PhysicsBlock>>&
    physicsBlocks() const;
    int integrationOrder() const;
    Teuchos::RCP<panzer::GlobalIndexer> dofManager() const;
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits>>
    linearObjectFactory() const;
    Teuchos::RCP<panzer::WorksetContainer> worksetContainer() const;
    const std::vector<panzer::BC>& boundaryConditions() const;
    Teuchos::RCP<panzer::BCStrategyFactory> boundaryConditionFactory() const;
    Teuchos::RCP<panzer::ClosureModelFactory_TemplateManager<panzer::Traits>>
    closureModelFactory() const;
    Teuchos::RCP<panzer::ModelEvaluator<double>> modelEvaluator() const;

  private:
    Teuchos::RCP<Parameter::ParameterDatabase> _parameter_db;
    Teuchos::RCP<MeshManager> _mesh_manager;
    double _t_init;
    Teuchos::RCP<panzer::GlobalData> _global_data;
    Teuchos::RCP<panzer::EquationSet_FactoryComposite> _eq_set_factory;
    std::vector<Teuchos::RCP<panzer::PhysicsBlock>> _physics_blocks;
    int _integration_order;
    Teuchos::RCP<panzer::GlobalIndexer> _dof_manager;
    Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits>> _linear_object_factory;
    Teuchos::RCP<panzer::WorksetContainer> _workset_container;
    std::vector<panzer::BC> _boundary_conditions;
    Teuchos::RCP<panzer::BCStrategyFactory> _bc_factory;
    Teuchos::RCP<panzer::BCFactoryComposite> _bc_factory_comp;
    Teuchos::RCP<panzer::ClosureModelFactory_TemplateManager<panzer::Traits>>
        _cm_factory;
    Teuchos::RCP<panzer::ModelEvaluator<double>> _model_evaluator;
    std::unordered_map<std::string, int> _parameter_indices;
};

} // end namespace VertexCFD

#endif // end VERTEXCFD_PHYSICSMANAGER_HPP
