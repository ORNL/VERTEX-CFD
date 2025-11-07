#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Panzer_BasisIRLayout.hpp>
#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <exception>
#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Workset Data
template<class EvalType>
struct TestData : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                  public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Dim> _node_coords;
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Point> _basis;
    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Point, panzer::Dim>
        _grad_basis;

    TestData(const panzer::IntegrationRule& ir,
             const panzer::BasisIRLayout& layout)
        : _node_coords("node_coords", layout.functional_grad)
        , _ip_coords("ip_coords", ir.dl_vector)
        , _basis("basis", layout.basis)
        , _grad_basis("grad_basis", layout.basis_grad)
    {
        this->addEvaluatedField(_node_coords);
        this->addEvaluatedField(_ip_coords);
        this->addEvaluatedField(_basis);
        this->addEvaluatedField(_grad_basis);
        this->setName("TestData");
    }

    void evaluateFields(typename panzer::Traits::EvalData workset) override
    {
        _node_coords.deep_copy(workset.int_rules[0]->node_coordinates);
        _ip_coords.deep_copy(workset.int_rules[0]->ip_coordinates);
        _basis.deep_copy(this->wda(workset).bases[0]->basis_scalar);
        _grad_basis.deep_copy(this->wda(workset).bases[0]->grad_basis);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void test2D_default_cell()
{
    // Setup test fixture.
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    EXPECT_STREQ("Quadrilateral_4", test_fixture.cell_topo->getName());

    // Create data.
    auto test_eval = Teuchos::rcp(new TestData<EvalType>(
        *test_fixture.ir, *test_fixture.basis_ir_layout));
    test_fixture.registerEvaluator<EvalType>(test_eval);

    // Set the time.
    EXPECT_EQ(0.0, test_fixture.workset->time);
    const double time = 1.3;
    test_fixture.setTime(time);
    EXPECT_EQ(time, test_fixture.workset->time);

    // Add required test fields. Check that some integration values were made
    // as well as some basis values.
    test_fixture.registerTestField<EvalType>(test_eval->_node_coords);
    test_fixture.registerTestField<EvalType>(test_eval->_ip_coords);
    test_fixture.registerTestField<EvalType>(test_eval->_basis);
    test_fixture.registerTestField<EvalType>(test_eval->_grad_basis);

    // Evaluate fields.
    test_fixture.evaluate<EvalType>();

    // Get fields.
    auto node_coords_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_node_coords);
    auto ip_coords_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_ip_coords);
    auto basis_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_basis);
    auto grad_basis_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_grad_basis);

    // Check fields.
    EXPECT_EQ(1, node_coords_result.extent(0));
    EXPECT_EQ(4, node_coords_result.extent(1));
    EXPECT_EQ(2, node_coords_result.extent(2));
    EXPECT_EQ(0.0, node_coords_result(0, 0, 0));
    EXPECT_EQ(0.0, node_coords_result(0, 0, 1));
    EXPECT_EQ(1.0, node_coords_result(0, 1, 0));
    EXPECT_EQ(0.0, node_coords_result(0, 1, 1));
    EXPECT_EQ(1.0, node_coords_result(0, 2, 0));
    EXPECT_EQ(1.0, node_coords_result(0, 2, 1));
    EXPECT_EQ(0.0, node_coords_result(0, 3, 0));
    EXPECT_EQ(1.0, node_coords_result(0, 3, 1));

    EXPECT_EQ(1, test_fixture.numPoint());
    EXPECT_EQ(1, ip_coords_result.extent(0));
    EXPECT_EQ(1, ip_coords_result.extent(1));
    EXPECT_EQ(2, ip_coords_result.extent(2));
    EXPECT_EQ(0.5, ip_coords_result(0, 0, 0));
    EXPECT_EQ(0.5, ip_coords_result(0, 0, 1));

    EXPECT_EQ(4, test_fixture.cardinality());
    EXPECT_EQ(1, basis_result.extent(0));
    EXPECT_EQ(4, basis_result.extent(1));
    EXPECT_EQ(1, basis_result.extent(2));
    EXPECT_EQ(0.25, basis_result(0, 0, 0));
    EXPECT_EQ(0.25, basis_result(0, 1, 0));
    EXPECT_EQ(0.25, basis_result(0, 2, 0));
    EXPECT_EQ(0.25, basis_result(0, 3, 0));

    EXPECT_EQ(1, grad_basis_result.extent(0));
    EXPECT_EQ(4, grad_basis_result.extent(1));
    EXPECT_EQ(1, grad_basis_result.extent(2));
    EXPECT_EQ(2, grad_basis_result.extent(3));
    EXPECT_EQ(-0.5, grad_basis_result(0, 0, 0, 0));
    EXPECT_EQ(-0.5, grad_basis_result(0, 0, 0, 1));
    EXPECT_EQ(0.5, grad_basis_result(0, 1, 0, 0));
    EXPECT_EQ(-0.5, grad_basis_result(0, 1, 0, 1));
    EXPECT_EQ(0.5, grad_basis_result(0, 2, 0, 0));
    EXPECT_EQ(0.5, grad_basis_result(0, 2, 0, 1));
    EXPECT_EQ(-0.5, grad_basis_result(0, 3, 0, 0));
    EXPECT_EQ(0.5, grad_basis_result(0, 3, 0, 1));
}

//---------------------------------------------------------------------------//
template<class EvalType>
void test3D_default_cell()
{
    // Setup test fixture.
    const int num_space_dim = 3;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    EXPECT_STREQ("Hexahedron_8", test_fixture.cell_topo->getName());

    // Create data.
    auto test_eval = Teuchos::rcp(new TestData<EvalType>(
        *test_fixture.ir, *test_fixture.basis_ir_layout));
    test_fixture.registerEvaluator<EvalType>(test_eval);

    // Set the time.
    EXPECT_EQ(0.0, test_fixture.workset->time);
    const double time = 1.3;
    test_fixture.setTime(time);
    EXPECT_EQ(time, test_fixture.workset->time);

    // Add required test fields. Check that some integration values were made
    // as well as some basis values.
    test_fixture.registerTestField<EvalType>(test_eval->_node_coords);
    test_fixture.registerTestField<EvalType>(test_eval->_ip_coords);
    test_fixture.registerTestField<EvalType>(test_eval->_basis);
    test_fixture.registerTestField<EvalType>(test_eval->_grad_basis);

    // Evaluate fields.
    test_fixture.evaluate<EvalType>();

    // Get fields.
    auto node_coords_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_node_coords);
    auto ip_coords_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_ip_coords);
    auto basis_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_basis);
    auto grad_basis_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_grad_basis);

    // Check fields.
    EXPECT_EQ(1, node_coords_result.extent(0));
    EXPECT_EQ(8, node_coords_result.extent(1));
    EXPECT_EQ(3, node_coords_result.extent(2));
    EXPECT_EQ(0.0, node_coords_result(0, 0, 0));
    EXPECT_EQ(0.0, node_coords_result(0, 0, 1));
    EXPECT_EQ(0.0, node_coords_result(0, 0, 2));
    EXPECT_EQ(1.0, node_coords_result(0, 1, 0));
    EXPECT_EQ(0.0, node_coords_result(0, 1, 1));
    EXPECT_EQ(0.0, node_coords_result(0, 1, 2));
    EXPECT_EQ(1.0, node_coords_result(0, 2, 0));
    EXPECT_EQ(1.0, node_coords_result(0, 2, 1));
    EXPECT_EQ(0.0, node_coords_result(0, 2, 2));
    EXPECT_EQ(0.0, node_coords_result(0, 3, 0));
    EXPECT_EQ(1.0, node_coords_result(0, 3, 1));
    EXPECT_EQ(0.0, node_coords_result(0, 3, 2));
    EXPECT_EQ(0.0, node_coords_result(0, 4, 0));
    EXPECT_EQ(0.0, node_coords_result(0, 4, 1));
    EXPECT_EQ(1.0, node_coords_result(0, 4, 2));
    EXPECT_EQ(1.0, node_coords_result(0, 5, 0));
    EXPECT_EQ(0.0, node_coords_result(0, 5, 1));
    EXPECT_EQ(1.0, node_coords_result(0, 5, 2));
    EXPECT_EQ(1.0, node_coords_result(0, 6, 0));
    EXPECT_EQ(1.0, node_coords_result(0, 6, 1));
    EXPECT_EQ(1.0, node_coords_result(0, 6, 2));
    EXPECT_EQ(0.0, node_coords_result(0, 7, 0));
    EXPECT_EQ(1.0, node_coords_result(0, 7, 1));
    EXPECT_EQ(1.0, node_coords_result(0, 7, 2));

    EXPECT_EQ(1, test_fixture.numPoint());
    EXPECT_EQ(1, ip_coords_result.extent(0));
    EXPECT_EQ(1, ip_coords_result.extent(1));
    EXPECT_EQ(3, ip_coords_result.extent(2));
    EXPECT_EQ(0.5, ip_coords_result(0, 0, 0));
    EXPECT_EQ(0.5, ip_coords_result(0, 0, 1));
    EXPECT_EQ(0.5, ip_coords_result(0, 0, 2));

    EXPECT_EQ(8, test_fixture.cardinality());
    EXPECT_EQ(1, basis_result.extent(0));
    EXPECT_EQ(8, basis_result.extent(1));
    EXPECT_EQ(1, basis_result.extent(2));
    EXPECT_EQ(0.125, basis_result(0, 0, 0));
    EXPECT_EQ(0.125, basis_result(0, 1, 0));
    EXPECT_EQ(0.125, basis_result(0, 2, 0));
    EXPECT_EQ(0.125, basis_result(0, 3, 0));
    EXPECT_EQ(0.125, basis_result(0, 4, 0));
    EXPECT_EQ(0.125, basis_result(0, 5, 0));
    EXPECT_EQ(0.125, basis_result(0, 6, 0));
    EXPECT_EQ(0.125, basis_result(0, 7, 0));

    EXPECT_EQ(1, grad_basis_result.extent(0));
    EXPECT_EQ(8, grad_basis_result.extent(1));
    EXPECT_EQ(1, grad_basis_result.extent(2));
    EXPECT_EQ(3, grad_basis_result.extent(3));
    EXPECT_EQ(-0.25, grad_basis_result(0, 0, 0, 0));
    EXPECT_EQ(-0.25, grad_basis_result(0, 0, 0, 1));
    EXPECT_EQ(-0.25, grad_basis_result(0, 0, 0, 2));
    EXPECT_EQ(0.25, grad_basis_result(0, 1, 0, 0));
    EXPECT_EQ(-0.25, grad_basis_result(0, 1, 0, 1));
    EXPECT_EQ(-0.25, grad_basis_result(0, 1, 0, 2));
    EXPECT_EQ(0.25, grad_basis_result(0, 2, 0, 0));
    EXPECT_EQ(0.25, grad_basis_result(0, 2, 0, 1));
    EXPECT_EQ(-0.25, grad_basis_result(0, 2, 0, 2));
    EXPECT_EQ(-0.25, grad_basis_result(0, 3, 0, 0));
    EXPECT_EQ(0.25, grad_basis_result(0, 3, 0, 1));
    EXPECT_EQ(-0.25, grad_basis_result(0, 3, 0, 2));
    EXPECT_EQ(-0.25, grad_basis_result(0, 4, 0, 0));
    EXPECT_EQ(-0.25, grad_basis_result(0, 4, 0, 1));
    EXPECT_EQ(0.25, grad_basis_result(0, 4, 0, 2));
    EXPECT_EQ(0.25, grad_basis_result(0, 5, 0, 0));
    EXPECT_EQ(-0.25, grad_basis_result(0, 5, 0, 1));
    EXPECT_EQ(0.25, grad_basis_result(0, 5, 0, 2));
    EXPECT_EQ(0.25, grad_basis_result(0, 6, 0, 0));
    EXPECT_EQ(0.25, grad_basis_result(0, 6, 0, 1));
    EXPECT_EQ(0.25, grad_basis_result(0, 6, 0, 2));
    EXPECT_EQ(-0.25, grad_basis_result(0, 7, 0, 0));
    EXPECT_EQ(0.25, grad_basis_result(0, 7, 0, 1));
    EXPECT_EQ(0.25, grad_basis_result(0, 7, 0, 2));
}

//---------------------------------------------------------------------------//
template<class EvalType>
void test_custom_cell()
{
    // Setup test fixture.
    const int num_cell = 1;
    const int nodes_per_cell = 3;
    const int num_space_dim = 2;

    const EvaluatorTestFixture::host_coords_view coords(
        "coords", num_cell, nodes_per_cell, num_space_dim);
    coords(0, 0, 0) = 0.0;
    coords(0, 0, 1) = 0.0;

    coords(0, 1, 0) = 1.0;
    coords(0, 1, 1) = 0.0;

    coords(0, 2, 0) = 0.0;
    coords(0, 2, 1) = 1.0;

    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        shards::getCellTopologyData<shards::Triangle<3>>(),
        coords,
        integration_order,
        basis_order);

    EXPECT_STREQ("Triangle_3", test_fixture.cell_topo->getName());

    // Create data.
    auto test_eval = Teuchos::rcp(new TestData<EvalType>(
        *test_fixture.ir, *test_fixture.basis_ir_layout));
    test_fixture.registerEvaluator<EvalType>(test_eval);

    // Set the time.
    EXPECT_EQ(0.0, test_fixture.workset->time);
    const double time = 1.3;
    test_fixture.setTime(time);
    EXPECT_EQ(time, test_fixture.workset->time);

    // Add required test fields. Check that some integration values were made
    // as well as some basis values.
    test_fixture.registerTestField<EvalType>(test_eval->_node_coords);
    test_fixture.registerTestField<EvalType>(test_eval->_ip_coords);
    test_fixture.registerTestField<EvalType>(test_eval->_basis);
    test_fixture.registerTestField<EvalType>(test_eval->_grad_basis);

    // Evaluate fields.
    test_fixture.evaluate<EvalType>();

    // Get fields.
    auto node_coords_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_node_coords);
    auto ip_coords_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_ip_coords);
    auto basis_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_basis);
    auto grad_basis_result
        = test_fixture.getTestFieldData<EvalType>(test_eval->_grad_basis);

    // Check fields.
    EXPECT_EQ(num_cell, node_coords_result.extent(0));
    EXPECT_EQ(nodes_per_cell, node_coords_result.extent(1));
    EXPECT_EQ(num_space_dim, node_coords_result.extent(2));
    for (int cell = 0; cell < num_cell; ++cell)
    {
        for (int node = 0; node < nodes_per_cell; ++node)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                EXPECT_EQ(coords(cell, node, dim),
                          node_coords_result(cell, node, dim));
            }
        }
    }

    EXPECT_EQ(1, test_fixture.numPoint());

    EXPECT_EQ(1, ip_coords_result.extent(0));
    EXPECT_EQ(1, ip_coords_result.extent(1));
    EXPECT_EQ(2, ip_coords_result.extent(2));
    EXPECT_DOUBLE_EQ(1.0 / 3.0, ip_coords_result(0, 0, 0));
    EXPECT_DOUBLE_EQ(1.0 / 3.0, ip_coords_result(0, 0, 1));

    EXPECT_EQ(3, test_fixture.cardinality());
    EXPECT_EQ(1, basis_result.extent(0));
    EXPECT_EQ(3, basis_result.extent(1));
    EXPECT_EQ(1, basis_result.extent(2));
    EXPECT_DOUBLE_EQ(1.0 / 3.0, basis_result(0, 0, 0));
    EXPECT_DOUBLE_EQ(1.0 / 3.0, basis_result(0, 1, 0));
    EXPECT_DOUBLE_EQ(1.0 / 3.0, basis_result(0, 2, 0));

    EXPECT_EQ(1, grad_basis_result.extent(0));
    EXPECT_EQ(3, grad_basis_result.extent(1));
    EXPECT_EQ(1, grad_basis_result.extent(2));
    EXPECT_EQ(2, grad_basis_result.extent(3));
    EXPECT_EQ(-1.0, grad_basis_result(0, 0, 0, 0));
    EXPECT_EQ(-1.0, grad_basis_result(0, 0, 0, 1));
    EXPECT_EQ(1.0, grad_basis_result(0, 1, 0, 0));
    EXPECT_EQ(0.0, grad_basis_result(0, 1, 0, 1));
    EXPECT_EQ(0.0, grad_basis_result(0, 2, 0, 0));
    EXPECT_EQ(1.0, grad_basis_result(0, 2, 0, 1));
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, residual_test_2d_default_cell)
{
    test2D_default_cell<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, jacobian_test_2d_default_cell)
{
    test2D_default_cell<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, residual_test_3d_default_cell)
{
    test3D_default_cell<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, jacobian_test_3d_default_cell)
{
    test3D_default_cell<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, residual_test_custom_cell)
{
    test_custom_cell<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, jacobian_test_custom_cell)
{
    test_custom_cell<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, default_cell_bad_space_dim)
{
    // Setup test fixture.
    const int num_space_dim = 4;
    const int integration_order = 1;
    const int basis_order = 1;

    const std::string msg
        = "Invalid spatial dimensions (4): must be 1, 2, or 3.";
    EXPECT_THROW(
        try {
            const EvaluatorTestFixture test_fixture(
                num_space_dim, integration_order, basis_order);
        } catch (const std::logic_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::logic_error);
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, custom_cell_bad_space_dim)
{
    // Setup test fixture.
    const int num_cell = 1;
    const int nodes_per_cell = 3;
    const int num_space_dim = 3;
    const EvaluatorTestFixture::host_coords_view coords(
        "coords", num_cell, nodes_per_cell, num_space_dim);
    const int integration_order = 1;
    const int basis_order = 1;

    const std::string msg
        = "Unexpected spatial dimensions in provided coordinates: "
          "Triangle_3 expects 2 dimensions, but 3 were provided.";
    EXPECT_THROW(
        try {
            const EvaluatorTestFixture test_fixture(
                shards::getCellTopologyData<shards::Triangle<3>>(),
                coords,
                integration_order,
                basis_order);
        } catch (const std::logic_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::logic_error);
}

//---------------------------------------------------------------------------//
TEST(EvaluatorTestHarness, custom_cell_bad_node_count)
{
    // Setup test fixture.
    const int num_cell = 1;
    const int nodes_per_cell = 4;
    const int num_space_dim = 2;
    const EvaluatorTestFixture::host_coords_view coords(
        "coords", num_cell, nodes_per_cell, num_space_dim);
    const int integration_order = 1;
    const int basis_order = 1;

    const std::string msg
        = "Unexpected node count in provided coordinates: "
          "Triangle_3 expects 3 nodes, but 4 were provided.";
    EXPECT_THROW(
        try {
            const EvaluatorTestFixture test_fixture(
                shards::getCellTopologyData<shards::Triangle<3>>(),
                coords,
                integration_order,
                basis_order);
        } catch (const std::logic_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::logic_error);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
