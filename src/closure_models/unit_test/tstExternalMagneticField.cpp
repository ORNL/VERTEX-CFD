#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

enum ExtMagnType
{
    constant,
    toroidal,
    x_tanh
};

template<class EvalType>
void testEval(const int num_space_dim,
              const ExtMagnType ext_magn_type = ExtMagnType::constant)
{
    // Test fixture
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set non-trivial quadrature points to avoid x = y
    if (ext_magn_type != ExtMagnType::constant)
    {
        auto ip_coord_view
            = test_fixture.int_values->ip_coordinates.get_static_view();
        auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
        ip_coord_mirror(0, 0, 0) = 0.2;
        ip_coord_mirror(0, 0, 1) = 0.3;
        if (num_space_dim == 3)
            ip_coord_mirror(0, 0, 2) = 1.0;
        Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);
    }

    const auto& ir = *test_fixture.ir;

    // Initialize class object to test and expected values
    Teuchos::ParameterList ext_mag_field_param_list;
    double exp_bx = 0.0;
    double exp_by = 0.0;
    double exp_bz = 0.0;
    if (ext_magn_type == ExtMagnType::constant)
    {
        ext_mag_field_param_list.set("External Magnetic Field Type",
                                     "constant");
        Teuchos::Array<double> ext_magn_vct(3);
        exp_bx = 1.3;
        exp_by = 3.5;
        exp_bz = 2.4;
        ext_magn_vct[0] = exp_bx;
        ext_magn_vct[1] = exp_by;
        ext_magn_vct[2] = exp_bz;
        ext_mag_field_param_list.set("External Magnetic Field Value",
                                     ext_magn_vct);

        // Add a test for time dependent external magnetic field
        if (num_space_dim == 3)
        {
            const double time = 0.25;
            test_fixture.setTime(time);
            Teuchos::Array<double> dB0_dt(3);
            for (int d = 0; d < 3; ++d)
            {
                dB0_dt[d] = pow(-1.1, d) * (d + 0.2);
                ext_magn_vct[d] += dB0_dt[d] * time;
            }
            ext_mag_field_param_list.set(
                "External Magnetic Field Time Rate of Change", dB0_dt);
            exp_bx += dB0_dt[0] * time;
            exp_by += dB0_dt[1] * time;
            exp_bz += dB0_dt[2] * time;
        }
    }
    else if (ext_magn_type == ExtMagnType::toroidal)
    {
        ext_mag_field_param_list.set("External Magnetic Field Type",
                                     "toroidal");
        exp_bx = -3.0;
        exp_by = 2.0;
        ext_mag_field_param_list.set("Toroidal Field Magnitude", 1.3);
    }
    else if (ext_magn_type == ExtMagnType::x_tanh)
    {
        ext_mag_field_param_list.set("External Magnetic Field Type", "x_tanh");
        exp_bx = 0.35434369377420455;
        exp_by = 0.7086873875484091;
        exp_bz = 1.0630310813226136;
        Teuchos::Array<double> ext_magn_vct(3);
        ext_magn_vct[0] = 1.0;
        ext_magn_vct[1] = 2.0;
        ext_magn_vct[2] = 3.0;
        ext_mag_field_param_list.set("External Magnetic Field Value",
                                     ext_magn_vct);
        ext_mag_field_param_list.set(
            "External Magnetic Field Distribution Offset", 0.5);
        ext_mag_field_param_list.set(
            "External Magnetic Field Distribution Curvature Coefficient", 1.0);
    }

    const auto eval = Teuchos::rcp(
        new ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>(
            ir, ext_mag_field_param_list));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < 3; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_ext_magn_field[dim]);

    test_fixture.evaluate<EvalType>();

    const auto ext_magn_field_0
        = test_fixture.getTestFieldData<EvalType>(eval->_ext_magn_field[0]);
    const auto ext_magn_field_1
        = test_fixture.getTestFieldData<EvalType>(eval->_ext_magn_field[1]);
    const auto ext_magn_field_2
        = test_fixture.getTestFieldData<EvalType>(eval->_ext_magn_field[2]);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_bx, fieldValue(ext_magn_field_0, 0, qp));
        EXPECT_EQ(exp_by, fieldValue(ext_magn_field_1, 0, qp));
        EXPECT_EQ(exp_bz, fieldValue(ext_magn_field_2, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConstantExternalMagneticField2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ConstantExternalMagneticField2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(ToroidalExternalMagneticField2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, ExtMagnType::toroidal);
}

//-----------------------------------------------------------------//
TEST(ToroidalExternalMagneticField2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, ExtMagnType::toroidal);
}

//-----------------------------------------------------------------//
TEST(TimeDependentExternalMagneticField3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(TimeDependentExternalMagneticField3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//-----------------------------------------------------------------//
TEST(XDependentExternalMagneticField3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, ExtMagnType::x_tanh);
}

//-----------------------------------------------------------------//
TEST(XDependentExternalMagneticField3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, ExtMagnType::x_tanh);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    const Teuchos::Array<double> ext_magn_vct(3);
    test_fixture.user_params.sublist("External Magnetic Field Parameters")
        .set("External Magnetic Field Value", ext_magn_vct);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "ExternalMagneticField";
    test_fixture.eval_name = "External Magnetic Field";
    test_fixture.template buildAndTest<
        ClosureModel::ExternalMagneticField<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(ExternalMagneticField_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(ExternalMagneticField_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
