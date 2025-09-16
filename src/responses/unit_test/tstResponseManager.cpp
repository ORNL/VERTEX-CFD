#include "VertexCFD_ResponseUnitTestConfig.hpp"

#include "drivers/VertexCFD_InitialConditionManager.hpp"
#include "drivers/VertexCFD_MeshManager.hpp"
#include "drivers/VertexCFD_PhysicsManager.hpp"
#include "parameters/VertexCFD_ParameterDatabase.hpp"
#include "responses/VertexCFD_ResponseManager.hpp"

#include <Panzer_ParameterLibraryUtilities.hpp>
#include <Panzer_Traits.hpp>

#include <Sacado_Traits.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <array>
#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{

// Some response evaluators currently run on the host only, so check if the
// current memory space is accessible on host. Unsupported response types will
// be avoided below.
constexpr bool host_accessible
    = Kokkos::SpaceAccessibility<Kokkos::HostSpace, PHX::MemSpace>::accessible;

//---------------------------------------------------------------------------//
template<typename EvalType>
void testScalarParameters(const bool before,
                          const Teuchos::RCP<PhysicsManager>& physics_manager)
{
    using scalar_type = typename EvalType::ScalarT;
    auto pl = physics_manager->globalData()->pl;

    auto get_scalar_value = [=](const std::string& name) {
        return Sacado::ScalarValue<scalar_type>::eval(
            panzer::accessScalarParameter<EvalType>(name, *pl)->getValue());
    };

    if (before)
    {
        EXPECT_DOUBLE_EQ(0.0, get_scalar_value("u integral"));
        EXPECT_DOUBLE_EQ(0.0, get_scalar_value("v integral"));
        EXPECT_DOUBLE_EQ(0.0, get_scalar_value("u surface integral"));
        EXPECT_DOUBLE_EQ(0.0, get_scalar_value("v surface integral"));
        if constexpr (host_accessible)
        {
            EXPECT_DOUBLE_EQ(0.0, get_scalar_value("u min"));
            EXPECT_DOUBLE_EQ(0.0, get_scalar_value("u max"));
            EXPECT_DOUBLE_EQ(0.0, get_scalar_value("v min"));
            EXPECT_DOUBLE_EQ(0.0, get_scalar_value("v max"));
            EXPECT_DOUBLE_EQ(0.0, get_scalar_value("u probe"));
            EXPECT_DOUBLE_EQ(0.0, get_scalar_value("v probe"));
        }
    }

    else
    {
        EXPECT_DOUBLE_EQ(4.0, get_scalar_value("u integral"));
        EXPECT_DOUBLE_EQ(6.0, get_scalar_value("v integral"));
        EXPECT_DOUBLE_EQ(12.0, get_scalar_value("u surface integral"));
        EXPECT_DOUBLE_EQ(18.0, get_scalar_value("v surface integral"));
        if constexpr (host_accessible)
        {
            EXPECT_DOUBLE_EQ(2.0, get_scalar_value("u min"));
            EXPECT_DOUBLE_EQ(2.0, get_scalar_value("u max"));
            EXPECT_DOUBLE_EQ(3.0, get_scalar_value("v min"));
            EXPECT_DOUBLE_EQ(3.0, get_scalar_value("v max"));
            EXPECT_DOUBLE_EQ(2.0, get_scalar_value("u probe"));
            EXPECT_DOUBLE_EQ(3.0, get_scalar_value("v probe"));
        }
    }
}

//---------------------------------------------------------------------------//
namespace
{
// Helper to reduce boilerplate.
struct Helper
{
    Teuchos::RCP<Parameter::ParameterDatabase> parameter_db;
    Teuchos::RCP<PhysicsManager> physics_manager;

    Helper(const int basis_order)
    {
        auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
            Teuchos::DefaultComm<int>::getComm());

        const std::string location = VERTEXCFD_RESPONSE_TEST_DATA_DIR;
        const std::string file = "response_manager_test.xml";

        parameter_db = Teuchos::rcp(
            new Parameter::ParameterDatabase(comm, location + file));

        parameter_db->physicsParameters()
            ->sublist("FluidPhysicsBlock", true)
            .sublist("Data", true)
            .set("Basis Order", basis_order);

        auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));

        physics_manager = Teuchos::rcp(new PhysicsManager(
            std::integral_constant<int, 2>{}, parameter_db, mesh_manager));
    }

    auto getSolutionVectors()
    {
        // Finish physics.
        physics_manager->setupModel();

        InitialConditionManager ic_manager(parameter_db,
                                           physics_manager->meshManager());

        Teuchos::RCP<Thyra::VectorBase<double>> x;
        Teuchos::RCP<Thyra::VectorBase<double>> x_dot;
        ic_manager.applyInitialConditions(
            std::integral_constant<int, 2>{}, *physics_manager, x, x_dot);

        return std::array{x, x_dot};
    }
};
} // namespace

//---------------------------------------------------------------------------//
void testResponseManager(const int basis_order)
{
    Helper helper(basis_order);
    auto& physics_manager = helper.physics_manager;

    Response::ResponseManager response_manager(physics_manager);

    // Volume responses over all element blocks.
    // Surface responses over all sidesets.
    std::vector<panzer::WorksetDescriptor> sideset_descriptors = {
        {"eblock-0_0", "top"},
        {"eblock-0_0", "bottom"},
        {"eblock-0_0", "left"},
        {"eblock-0_0", "right"},
    };

    response_manager.addFunctionalResponse("u integral", "velocity_0");
    response_manager.addFunctionalResponse("v integral", "velocity_1");
    response_manager.addFunctionalResponse(
        "u surface integral", "velocity_0", sideset_descriptors);
    response_manager.addFunctionalResponse(
        "v surface integral", "velocity_1", sideset_descriptors);
    if constexpr (host_accessible)
    {
        response_manager.addMinValueResponse("u min", "velocity_0");
        response_manager.addMaxValueResponse("u max", "velocity_0");
        response_manager.addMinValueResponse("v min", "velocity_1");
        response_manager.addMaxValueResponse("v max", "velocity_1");
        Teuchos::Array<double> point(2);
        point[0] = 0.21875;
        point[1] = 1.5625;
        response_manager.addProbeResponse("u probe", "velocity_0", point);
        response_manager.addProbeResponse("v probe", "velocity_1", point);
    }

    // We will test responses of velocities, which are linear in (x,y):
    //   u = -0.25 * y
    //   v =  0.25 * x

    constexpr int num_resp = host_accessible ? 10 : 4;

    EXPECT_EQ(num_resp, response_manager.numResponses());

    for (int i = 0; i < num_resp; ++i)
        EXPECT_EQ(i, response_manager.globalIndex(i));

    EXPECT_EQ(0, response_manager.globalIndex("u integral"));
    EXPECT_EQ(1, response_manager.globalIndex("v integral"));
    EXPECT_EQ(2, response_manager.globalIndex("u surface integral"));
    EXPECT_EQ(3, response_manager.globalIndex("v surface integral"));
    if constexpr (host_accessible)
    {
        EXPECT_EQ(4, response_manager.globalIndex("u min"));
        EXPECT_EQ(5, response_manager.globalIndex("u max"));
        EXPECT_EQ(6, response_manager.globalIndex("v min"));
        EXPECT_EQ(7, response_manager.globalIndex("v max"));
        EXPECT_EQ(8, response_manager.globalIndex("u probe"));
        EXPECT_EQ(9, response_manager.globalIndex("v probe"));
    }

    EXPECT_EQ("u integral", response_manager.name(0));
    EXPECT_EQ("v integral", response_manager.name(1));
    EXPECT_EQ("u surface integral", response_manager.name(2));
    EXPECT_EQ("v surface integral", response_manager.name(3));
    if constexpr (host_accessible)
    {
        EXPECT_EQ("u min", response_manager.name(4));
        EXPECT_EQ("u max", response_manager.name(5));
        EXPECT_EQ("v min", response_manager.name(6));
        EXPECT_EQ("v max", response_manager.name(7));
        EXPECT_EQ("u probe", response_manager.name(8));
        EXPECT_EQ("v probe", response_manager.name(9));
    }

    testScalarParameters<panzer::Traits::Residual>(true, physics_manager);
    testScalarParameters<panzer::Traits::Jacobian>(true, physics_manager);
    testScalarParameters<panzer::Traits::Tangent>(true, physics_manager);

    auto [x, x_dot] = helper.getSolutionVectors();
    response_manager.evaluateResponses(x, x_dot, 0.0, 0.0);

    EXPECT_DOUBLE_EQ(4.0, response_manager.value(0));
    EXPECT_DOUBLE_EQ(6.0, response_manager.value(1));
    EXPECT_DOUBLE_EQ(12.0, response_manager.value(2));
    EXPECT_DOUBLE_EQ(18.0, response_manager.value(3));
    if constexpr (host_accessible)
    {
        EXPECT_DOUBLE_EQ(2.0, response_manager.value(4));
        EXPECT_DOUBLE_EQ(2.0, response_manager.value(5));
        EXPECT_DOUBLE_EQ(3.0, response_manager.value(6));
        EXPECT_DOUBLE_EQ(3.0, response_manager.value(7));
        EXPECT_DOUBLE_EQ(2.0, response_manager.value(8));
        EXPECT_DOUBLE_EQ(3.0, response_manager.value(9));
    }

    EXPECT_DOUBLE_EQ(4.0, response_manager.value("u integral"));
    EXPECT_DOUBLE_EQ(6.0, response_manager.value("v integral"));
    EXPECT_DOUBLE_EQ(12.0, response_manager.value("u surface integral"));
    EXPECT_DOUBLE_EQ(18.0, response_manager.value("v surface integral"));
    if constexpr (host_accessible)
    {
        EXPECT_DOUBLE_EQ(2.0, response_manager.value("u min"));
        EXPECT_DOUBLE_EQ(2.0, response_manager.value("u max"));
        EXPECT_DOUBLE_EQ(3.0, response_manager.value("v min"));
        EXPECT_DOUBLE_EQ(3.0, response_manager.value("v max"));
        EXPECT_DOUBLE_EQ(2.0, response_manager.value("u probe"));
        EXPECT_DOUBLE_EQ(3.0, response_manager.value("v probe"));
    }

    testScalarParameters<panzer::Traits::Residual>(false, physics_manager);
    testScalarParameters<panzer::Traits::Jacobian>(false, physics_manager);
    testScalarParameters<panzer::Traits::Tangent>(false, physics_manager);
}

//---------------------------------------------------------------------------//
TEST(ResponseManager, BasisOrder1)
{
    testResponseManager(1);
}

//---------------------------------------------------------------------------//
TEST(ResponseManager, BasisOrder2)
{
    testResponseManager(2);
}

//---------------------------------------------------------------------------//
// Test against a bug in the internal name to index mapping that was returning
// the global rather than local index.
TEST(MultipleResponseManagers, NameToIndexBug)
{
    Helper helper(1);
    auto& physics_manager = helper.physics_manager;

    Response::ResponseManager rm1(physics_manager);
    Response::ResponseManager rm2(physics_manager);

    rm1.addFunctionalResponse("u integral", "velocity_0");
    rm2.addFunctionalResponse("v integral", "velocity_1");

    int u_index = -1;
    EXPECT_NO_THROW(u_index = rm1.globalIndex("u integral"));
    EXPECT_EQ(0, u_index);

    // The bug causes out of range access which throws.
    int v_index = -1;
    EXPECT_NO_THROW(v_index = rm2.globalIndex("v integral"));
    EXPECT_EQ(1, v_index);

    auto [x, x_dot] = helper.getSolutionVectors();
    rm1.evaluateResponses(x, x_dot, 0.0, 0.0);
    rm2.evaluateResponses(x, x_dot, 0.0, 0.0);

    double u_integral = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NO_THROW(u_integral = rm1.value("u integral"));
    EXPECT_DOUBLE_EQ(4.0, u_integral);

    // The bug causes out of range access which throws.
    double v_integral = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NO_THROW(v_integral = rm2.value("v integral"));
    EXPECT_DOUBLE_EQ(6.0, v_integral);
}

//---------------------------------------------------------------------------//
// Test against a response name lookup bug in which the local response index
// was being used to get the name from the global response library.
TEST(MultipleResponseManagers, NameLookupBug)
{
    Helper helper(1);
    auto& physics_manager = helper.physics_manager;

    Response::ResponseManager rm1(physics_manager);
    Response::ResponseManager rm2(physics_manager);

    rm1.addFunctionalResponse("u integral", "velocity_0");
    rm2.addFunctionalResponse("v integral", "velocity_1");

    // The bug causes both responses to return the same name.
    EXPECT_EQ("u integral", rm1.name(0));
    EXPECT_EQ("v integral", rm2.name(0));

    auto pl = physics_manager->globalData()->pl;
    using Eval = panzer::Traits::Residual;

    EXPECT_EQ(0.0, pl->getRealValue<Eval>("u integral"));
    EXPECT_EQ(0.0, pl->getRealValue<Eval>("v integral"));

    // Evaluate both response managers, which will also set values in the
    // parameter library.
    auto [x, x_dot] = helper.getSolutionVectors();
    rm1.evaluateResponses(x, x_dot, 0.0, 0.0);
    rm2.evaluateResponses(x, x_dot, 0.0, 0.0);

    // With the bug, the "u_integral" parameter gets overwritten and
    // "v_integral" is never set.
    EXPECT_DOUBLE_EQ(4.0, pl->getRealValue<Eval>("u integral"));
    EXPECT_DOUBLE_EQ(6.0, pl->getRealValue<Eval>("v integral"));
}

} // namespace Test
} // namespace VertexCFD
