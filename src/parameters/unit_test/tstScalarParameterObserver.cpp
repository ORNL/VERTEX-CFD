#include "VertexCFD_ParameterUnitTestConfig.hpp"

#include "parameters/VertexCFD_GeneralScalarParameterInput.hpp"
#include "parameters/VertexCFD_ParameterDatabase.hpp"
#include "parameters/VertexCFD_ScalarParameterInput.hpp"
#include "parameters/VertexCFD_ScalarParameterObserver.hpp"

#include "drivers/VertexCFD_InitialConditionManager.hpp"
#include "drivers/VertexCFD_MeshManager.hpp"
#include "drivers/VertexCFD_PhysicsManager.hpp"

#include <Panzer_Workset.hpp>

#include <gtest/gtest.h>

#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
class TestObserver : public Parameter::ScalarParameterObserver<EvalType>
{
  public:
    void updateStateWithNewParameters() override { _state_updated = true; }
    bool _state_updated = false;
};

//---------------------------------------------------------------------------//
double getValue(const panzer::Traits::Residual::ScalarT& observed)
{
    return observed;
}

//---------------------------------------------------------------------------//
double getValue(const panzer::Traits::Jacobian::ScalarT& observed)
{
    return observed.val();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testScalarParameterObserver()
{
    // Setup base data structures to make a parameter library.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());
    const std::string location = VERTEXCFD_RESPONSE_TEST_DATA_DIR;
    const std::string file = "response_manager_test.xml";
    auto parameter_db = Teuchos::rcp(
        new Parameter::ParameterDatabase(comm, location + file));
    parameter_db->physicsParameters()
        ->sublist("FluidPhysicsBlock", true)
        .sublist("Data", true)
        .set("Basis Order", 1);
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));
    auto physics_manager = Teuchos::rcp(new PhysicsManager(
        std::integral_constant<int, 2>{}, parameter_db, mesh_manager));
    physics_manager->setupModel();

    // Add a global parameter
    physics_manager->addScalarParameter("Global Parameter", 1.234);

    // Make worksets to get block info. First workset has a real block, second
    // does not so we can check default assignment.
    panzer::Workset workset_1;
    workset_1.block_id = "eblock-0_0";
    panzer::Workset workset_2;
    workset_2.block_id = "undefined";

    // Make general parameter data.
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
        general_parameter_data;
    std::unordered_map<std::string, double> param_values;
    param_values.emplace("Default Value", 5.678);
    param_values.emplace("eblock-0_0", 6.789);
    general_parameter_data.emplace("General Parameter", param_values);

    // Make parameter list.
    Teuchos::ParameterList plist;

    // Add a global parameter.
    const Parameter::ScalarParameterInput global_input = {"Global Parameter"};
    plist.set("global_param", global_input);

    // Add a general parameter.
    const Parameter::GeneralScalarParameterInput general_input
        = {"General Parameter"};
    plist.set("general_param", general_input);

    // Add a regular parameter.
    plist.set("local_param_1", 3.456);

    // Make parameters.
    typename EvalType::ScalarT global_param;
    typename EvalType::ScalarT general_param;
    typename EvalType::ScalarT local_param_1;
    typename EvalType::ScalarT local_param_2;

    // Make a parameter observer.
    TestObserver<EvalType> observer;
    observer.registerParameter("global_param", 2.345, plist, global_param);
    observer.registerParameter("general_param", 0.123, plist, general_param);
    observer.registerParameter("local_param_1", 4.567, plist, local_param_1);
    observer.registerParameter("local_param_2", 4.567, plist, local_param_2);

    // Check pre-update values.
    EXPECT_EQ(2.345, getValue(global_param));
    EXPECT_EQ(0.123, getValue(general_param));
    EXPECT_EQ(3.456, getValue(local_param_1));
    EXPECT_EQ(4.567, getValue(local_param_2));

    // Update with the first workset.
    EXPECT_FALSE(observer._state_updated);
    observer.update(
        *(physics_manager->globalData()), workset_1, general_parameter_data);
    EXPECT_TRUE(observer._state_updated);

    // Check post-update values.
    EXPECT_EQ(1.234, getValue(global_param));
    EXPECT_EQ(6.789, getValue(general_param));
    EXPECT_EQ(3.456, getValue(local_param_1));
    EXPECT_EQ(4.567, getValue(local_param_2));

    // Update with the second workset and make sure the block default was
    // given. The general param should be the only one that changes.
    observer.update(
        *(physics_manager->globalData()), workset_2, general_parameter_data);
    EXPECT_EQ(1.234, getValue(global_param));
    EXPECT_EQ(5.678, getValue(general_param));
    EXPECT_EQ(3.456, getValue(local_param_1));
    EXPECT_EQ(4.567, getValue(local_param_2));
}

//---------------------------------------------------------------------------//
TEST(ScalarParameterObserver, residual)
{
    testScalarParameterObserver<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ScalarParameterObserver, jacobian)
{
    testScalarParameterObserver<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
