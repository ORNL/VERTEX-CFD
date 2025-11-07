#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "induction_less_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_ElectricCurrentDensityInsulating.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test

{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    int num_grad_dim;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> bc_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> bc_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> bc_velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ext_magn_field_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ext_magn_field_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ext_magn_field_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : num_grad_dim(ir.spatial_dimension)
        , _electric_potential("electric_potential", ir.dl_scalar)
        , _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
        , bc_velocity_0("BOUNDARY_velocity_0", ir.dl_scalar)
        , bc_velocity_1("BOUNDARY_velocity_1", ir.dl_scalar)
        , bc_velocity_2("BOUNDARY_velocity_2", ir.dl_scalar)
        , velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , ext_magn_field_0("external_magnetic_field_0", ir.dl_scalar)
        , ext_magn_field_1("external_magnetic_field_1", ir.dl_scalar)
        , ext_magn_field_2("external_magnetic_field_2", ir.dl_scalar)
    {
        this->addEvaluatedField(_electric_potential);
        this->addEvaluatedField(_grad_electric_potential);
        this->addEvaluatedField(_normals);

        this->addEvaluatedField(bc_velocity_0);
        this->addEvaluatedField(bc_velocity_1);
        this->addEvaluatedField(bc_velocity_2);
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(ext_magn_field_0);
        this->addEvaluatedField(ext_magn_field_1);
        this->addEvaluatedField(ext_magn_field_2);
        this->setName(
            "Electric Current Density Insulating Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _electric_potential.deep_copy(2.0);

        Kokkos::parallel_for(
            "electric current density insulating test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _grad_electric_potential.extent(1);
        const int num_grad_dim = _grad_electric_potential.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_grad_dim; ++d)
            {
                const int dqp = (qp + 1) * (d + 1);
                _grad_electric_potential(c, qp, d) = dqp * 3.0;
                _normals(c, qp, d) = dqp * 0.1;
            }

            bc_velocity_0(c, qp) = 0.6;
            velocity_0(c, qp) = 0.5;
            bc_velocity_1(c, qp) = 1.4;
            velocity_1(c, qp) = 1.5;
            if (num_grad_dim == 3)
            {
                bc_velocity_2(c, qp) = 3.0;
                velocity_2(c, qp) = 2.5;
                ext_magn_field_0(c, qp) = 1.1;
                ext_magn_field_1(c, qp) = 2.0;
                ext_magn_field_2(c, qp) = -0.3;
            }
            else
            {
                bc_velocity_2(c, qp) = _nanval;
                velocity_2(c, qp) = _nanval;
                ext_magn_field_0(c, qp) = 0.0;
                ext_magn_field_1(c, qp) = 0.0;
                ext_magn_field_2(c, qp) = 0.3;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create evaluator.
    const auto fixed_eval = Teuchos::rcp(
        new BoundaryCondition::ElectricCurrentDensityInsulating<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(fixed_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_electric_potential);
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_grad_electric_potential);

    // Evaluate variables.
    test_fixture.evaluate<EvalType>();

    // Check values.
    const auto boundary_ep_result = test_fixture.getTestFieldData<EvalType>(
        fixed_eval->_boundary_electric_potential);
    const auto boundary_grad_ep_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_electric_potential);

    const double exp_values_2d[3] = {2.8815, 5.733, _nanval};
    const double exp_values_3d[3] = {3.534, 4.548, 7.382};

    const double* exp_values
        = (num_space_dim == 3 ? exp_values_3d : exp_values_2d);

    const int num_point = boundary_ep_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(2.0, fieldValue(boundary_ep_result, 0, qp));

        for (int d = 0; d < num_space_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(exp_values[d],
                             fieldValue(boundary_grad_ep_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D residual
TEST(ElectricCurrentDensityInsulating2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

// 2-D jacobian
TEST(ElectricCurrentDensityInsulating2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
// 3-D residual
TEST(ElectricCurrentDensityInsulating3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

// 3-D jacobian
TEST(ElectricCurrentDensityInsulating3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
