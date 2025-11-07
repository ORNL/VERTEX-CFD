#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include <closure_models/VertexCFD_Closure_WallDistance.hpp>
#include <drivers/VertexCFD_MeshManager.hpp>

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct ExpectedDistance
    : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    int _num_space_dim;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _exp_distance;
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

    int _ir_degree;

    ExpectedDistance(const panzer::IntegrationRule& ir)
        : _num_space_dim(ir.spatial_dimension)
        , _exp_distance("exp_distance", ir.dl_scalar)
        , _ir_degree(ir.cubature_degree)
    {
        this->addEvaluatedField(_exp_distance);

        this->setName("Wall Distance Unit Test Expected Distance");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        auto _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, d);
        _ip_coords = d.int_rules[_ir_index]->ip_coordinates;
        Kokkos::parallel_for(
            "Wall Distance Unit Test Expected Distance",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _exp_distance.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            const double y_dist = 1 - std::abs(_ip_coords(c, qp, 1));
            double z_dist = 1e8;
            if (_num_space_dim == 3)
            {
                z_dist = 1 - std::abs(_ip_coords(c, qp, 2));
            }
            _exp_distance(c, qp) = std::fmin(y_dist, z_dist);
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const std::string element_type)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Construct Mesh parameters and manager
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Make an empty parameter database and build mesh parameters.
    const Parameter::ParameterDatabase parameter_db(comm);
    auto mesh_params = parameter_db.meshParameters();
    mesh_params->set("Mesh Input Type", "Inline");
    auto& inline_params = mesh_params->sublist("Inline");
    inline_params.set("Element Type", element_type);
    auto& mesh_details = inline_params.sublist("Mesh");
    const int nelem_x = 2;
    const int nelem_y = 2;
    const int nelem_z = 2;
    mesh_details.set("X0", 0.0);
    mesh_details.set("Xf", 1.0);
    mesh_details.set("X Elements", nelem_x);
    mesh_details.set("Y0", -1.0);
    mesh_details.set("Yf", 1.0);
    mesh_details.set("Y Elements", nelem_y);
    if (num_space_dim == 3)
    {
        mesh_details.set("Z0", -1.0);
        mesh_details.set("Zf", 1.0);
        mesh_details.set("Z Elements", nelem_z);
    }

    const Teuchos::RCP<MeshManager> mesh_manager{
        Teuchos::rcp(new MeshManager(parameter_db, comm))};
    mesh_manager->completeMeshConstruction();

    const auto sideset_geometry
        = Teuchos::rcp(new VertexCFD::Mesh::Topology::SidesetGeometry(
            mesh_manager->mesh(), {"top", "bottom", "front", "back"}));

    const auto exp_dist_eval = Teuchos::rcp(new ExpectedDistance<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(exp_dist_eval);
    test_fixture.registerTestField<EvalType>(exp_dist_eval->_exp_distance);

    // Initialize and register closure model
    auto eval = Teuchos::rcp(
        new ClosureModel::WallDistance<EvalType, panzer::Traits, num_space_dim>(
            ir, sideset_geometry));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_distance);

    test_fixture.evaluate<EvalType>();

    // Evaluate and assert closure model values
    auto fv_dist = test_fixture.getTestFieldData<EvalType>(eval->_distance);
    auto fv_exp_dist = test_fixture.getTestFieldData<EvalType>(
        exp_dist_eval->_exp_distance);

    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        // Wall distance
        EXPECT_EQ(fieldValue(fv_exp_dist, 0, qp), fieldValue(fv_dist, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(WallDistanceTet4, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("Tet4");
}

//-----------------------------------------------------------------//
TEST(WallDistanceTet4, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("Tet4");
}

//-----------------------------------------------------------------//
TEST(WallDistanceTri3, residual_test)
{
    testEval<panzer::Traits::Residual, 2>("Tri3");
}

//-----------------------------------------------------------------//
TEST(WallDistanceTri3, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>("Tri3");
}

//-----------------------------------------------------------------//
TEST(WallDistanceHex8, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("Hex8");
}

//-----------------------------------------------------------------//
TEST(WallDistanceHex8, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("Hex8");
}

//-----------------------------------------------------------------//
TEST(WallDistanceQuad4, residual_test)
{
    testEval<panzer::Traits::Residual, 2>("Quad4");
}

//-----------------------------------------------------------------//
TEST(WallDistanceQuad4, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>("Quad4");
}

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "WallDistance";
    test_fixture.eval_name = "distance";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);

    // Construct a minimal mesh needed for construction of the wall distance
    // closure model.
    {
        auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
            Teuchos::DefaultComm<int>::getComm());

        const Parameter::ParameterDatabase parameter_db(comm);
        parameter_db.meshParameters()
            ->set("Mesh Input Type", "Inline")
            .sublist("Inline")
            .set("Element Type", "Tet4")
            .sublist("Mesh")
            .set("X Elements", 1)
            .set("Y Elements", 1)
            .set("Z Elements", 1);
        auto mesh_manager{Teuchos::rcp(new MeshManager(parameter_db, comm))};
        mesh_manager->completeMeshConstruction();

        test_fixture.user_params.set("MeshManager", mesh_manager)
            .sublist("Wall List")
            .set("Wall Names", "top");
        const auto sideset_geometry
            = Teuchos::rcp(new VertexCFD::Mesh::Topology::SidesetGeometry(
                mesh_manager->mesh(), {"top"}));

        test_fixture.user_params.set("Sideset Geometry", sideset_geometry);
    }

    test_fixture.template buildAndTest<
        ClosureModel::WallDistance<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(WallDistance_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(WallDistance_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

TEST(WallDistance_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(WallDistance_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
