#ifndef VERTEXCFD_COMPRESSIBLEMHDCLOSUREMODELFACTORYTESTHARNESS_HPP
#define VERTEXCFD_COMPRESSIBLEMHDCLOSUREMODELFACTORYTESTHARNESS_HPP

#include "drivers/VertexCFD_MeshManager.hpp"

#include "plasma_solver/closure_models/VertexCFD_PlasmaSpeciesClosureModelFactory.hpp"

#include <Panzer_FieldLibrary.hpp>
#include <Panzer_IntegrationDescriptor.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_FieldManager.hpp>

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Closure model factory test fixture.
template<class EvalType>
struct PlasmaSpeciesClosureModelFactoryTestFixture
{
    int num_evaluators = 1;
    int eval_index = -1;
    std::string type_name{"!!! UNDEFINED !!!"};
    std::string eval_name{"!!! UNDEFINED !!!"};
    std::string factory_type = "Plasma Species MHD";
    Teuchos::ParameterList user_params{"User Data"};
    Teuchos::ParameterList model_params{"Model Data"};
    Teuchos::ParameterList closure_params{"Closure Models"};
    const std::string model_id{"Test Model"};

    PlasmaSpeciesClosureModelFactoryTestFixture()
    {
        auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
            Teuchos::DefaultComm<int>::getComm());
        const Parameter::ParameterDatabase parameter_db(comm);

        auto mesh_params = parameter_db.meshParameters();
        mesh_params->set("Mesh Input Type", "Inline");
        auto& inline_params = mesh_params->sublist("Inline");
        inline_params.set("Element Type", "Tet4");
        auto& mesh_details = inline_params.sublist("Mesh");
        mesh_details.set("X0", 0.0);
        mesh_details.set("Xf", 1.0);
        mesh_details.set("X Elements", 10);
        mesh_details.set("Y0", -1.0);
        mesh_details.set("Yf", 1.0);
        mesh_details.set("Y Elements", 10);
        mesh_details.set("Z0", -1.0);
        mesh_details.set("Zf", 1.0);
        mesh_details.set("Z Elements", 10);

        const Teuchos::RCP<MeshManager> mesh_manager{
            Teuchos::rcp(new MeshManager(parameter_db, comm))};
        user_params.set<const Teuchos::RCP<MeshManager>>("MeshManager",
                                                         mesh_manager);
        mesh_manager->completeMeshConstruction();
    }

    template<class Evaluator, int NumSpaceDim>
    void buildAndTest(const bool add_species_prop = true)
    {
        constexpr int num_space_dim = NumSpaceDim;

        // Set up closure parameter model
        model_params.set("Type", type_name);
        closure_params
            .sublist(model_id)                // Model ID sublist
            .set("Model Data", model_params); // sublist for Model Data
        closure_params.sublist(model_id).set("Closure Factory Type",
                                             factory_type);
        if (add_species_prop)
        {
            closure_params.sublist(model_id)
                .sublist("Species Properties")
                .sublist("electron")
                .set("Species Mass", 1.3);
            closure_params.sublist(model_id)
                .sublist("Species Properties")
                .sublist("neutron")
                .set("Species Mass", 1.4);
        }

        const int num_species = 2;

        // Set up a trivial integration_rule for the factory.
        const panzer::IntegrationDescriptor integration_desc(
            0, panzer::IntegrationDescriptor::VOLUME);
        auto cell_topo = Teuchos::rcp(new shards::CellTopology(
            shards::getCellTopologyData<shards::Quadrilateral<4>>()));
        const int num_cells = 0;
        auto integration_rule = Teuchos::rcp(new panzer::IntegrationRule(
            integration_desc, cell_topo, num_cells));

        // This is unused but passed by non-const reference, so cannot be
        // constructed in the argument list.
        PHX::FieldManager<panzer::Traits> fm;

        const ClosureModel::PlasmaSpeciesFactory<EvalType, num_space_dim>
            cmhd_factory{};
        auto evaluators = cmhd_factory.buildClosureModels(
            model_id,
            closure_params,
            panzer::FieldLayoutLibrary{}, // unused
            integration_rule,
            Teuchos::ParameterList{}, // unused (default_params)
            user_params,
            Teuchos::null, // unused (global_data)
            fm);

        // Make sure the factory returned something.
        ASSERT_FALSE(evaluators.is_null());

        // The factory should return the given number of evaluators.
        num_evaluators += cmhd_factory.numDefaultClosures(num_species);
        ASSERT_EQ(num_evaluators, evaluators->size());

        // Check evaluator at the given index
        auto eval = [&]() {
            if (eval_index < 0)
                return evaluators->back();
            else
                return evaluators->at(eval_index);
        }();

        // The evaluator should be castable to the derived evaluator type.
        auto evalPtr = dynamic_cast<Evaluator*>(eval.get());
        EXPECT_NE(nullptr, evalPtr) << "Factory returned unexpected "
                                       "evaluator.";

        // The evaluator should have the expected name.
        EXPECT_EQ(eval_name, eval->getName());
    }
};

} // namespace Test
} // namespace VertexCFD

#endif // VERTEXCFD_COMPRESSIBLEMHDCLOSUREMODELFACTORYTESTHARNESS_HPP
