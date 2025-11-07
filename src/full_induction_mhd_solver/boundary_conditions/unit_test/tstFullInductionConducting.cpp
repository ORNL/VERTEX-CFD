#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "full_induction_mhd_solver/boundary_conditions/VertexCFD_BoundaryState_FullInductionConducting.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType, int NumSpaceDim>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int num_magnetic_field_dim = 3;

    double _scalar_magn_pot;
    double _eta;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_velocity_2;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _induced_magnetic_field;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_induced_magnetic_field;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _scalar_magnetic_potential;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _resistivity;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_magnetic_field_dim>
        _external_magnetic_field;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double scalar_magn_pot,
                 const double eta)
        : _scalar_magn_pot(scalar_magn_pot)
        , _eta(eta)
        , _boundary_velocity_0("BOUNDARY_velocity_0", ir.dl_scalar)
        , _boundary_velocity_1("BOUNDARY_velocity_1", ir.dl_scalar)
        , _boundary_velocity_2("BOUNDARY_velocity_2", ir.dl_scalar)
        , _scalar_magnetic_potential("scalar_magnetic_potential", ir.dl_scalar)
        , _resistivity("resistivity", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_boundary_velocity_0);
        this->addEvaluatedField(_boundary_velocity_1);
        if (num_space_dim == 3)
            this->addEvaluatedField(_boundary_velocity_2);

        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_scalar,
                                       _induced_magnetic_field,
                                       "induced_magnetic_field_");

        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_vector,
                                       _grad_induced_magnetic_field,
                                       "GRAD_induced_magnetic_field_");

        this->addEvaluatedField(_scalar_magnetic_potential);
        this->addEvaluatedField(_resistivity);

        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_scalar,
                                       _external_magnetic_field,
                                       "external_magnetic_field_");

        this->addEvaluatedField(_normals);

        this->setName(
            "Full Induction Model Conducting Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _boundary_velocity_0.deep_copy(3.0);
        _boundary_velocity_1.deep_copy(-4.0);
        if (num_space_dim == 3)
            _boundary_velocity_2.deep_copy(5.0);

        for (int dim = 0; dim < num_magnetic_field_dim; ++dim)
        {
            if (dim < num_space_dim)
            {
                _induced_magnetic_field[dim].deep_copy(1.25 * (dim + 1));
            }
            _external_magnetic_field[dim].deep_copy(pow(-0.5, dim + 1) * 0.1);
        }

        _scalar_magnetic_potential.deep_copy(_scalar_magn_pot);
        _resistivity.deep_copy(_eta);

        Kokkos::parallel_for(
            "full induction model conducting unit test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _grad_induced_magnetic_field[0].extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            _normals(c, qp, 0) = 0.45;
            _normals(c, qp, 1) = -0.65;
            if (num_space_dim == 3)
                _normals(c, qp, 2) = 0.35;
            for (int fdim = 0; fdim < num_space_dim; ++fdim)
            {
                for (int gdim = 0; gdim < num_space_dim; ++gdim)
                {
                    _grad_induced_magnetic_field[fdim](c, qp, gdim)
                        = (fdim + 11) * pow(-0.5, fdim + gdim);
                }
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(bool build_magn_corr,
              bool dirichlet_scalar_magn_pot,
              bool build_resistive_flux)
{
    // Test fixture
    static constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const double nanval = std::numeric_limits<double>::signaling_NaN();
    const double scalar_magn_pot
        = (build_magn_corr && !dirichlet_scalar_magn_pot) ? 4.4 : nanval;
    const Kokkos::Array<double, 3> bnd_b = {1.1, 2.2, 3.3};
    const double eta = 3.6;

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(
            *test_fixture.ir, scalar_magn_pot, eta));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create conducting wall evaluator.
    const double bnd_scalar_magn_pot
        = (build_magn_corr && dirichlet_scalar_magn_pot) ? 5.5 : nanval;

    Teuchos::ParameterList bc_params;
    bc_params.set("induced_magnetic_field_0", bnd_b[0]);
    bc_params.set("induced_magnetic_field_1", bnd_b[1]);
    if (num_space_dim == 3)
        bc_params.set("induced_magnetic_field_2", bnd_b[2]);
    if (build_magn_corr && dirichlet_scalar_magn_pot)
        bc_params.set("scalar_magnetic_potential", bnd_scalar_magn_pot);

    Teuchos::ParameterList full_indu_params;
    full_indu_params.set("Vacuum Magnetic Permeability", 0.12)
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Hyperbolic Divergence Cleaning Speed", 1.1)
        .set("Build Resistive Flux", build_resistive_flux)
        .set("Resistivity", eta);
    const MHDProperties::FullInductionMHDProperties mhd_props(full_indu_params);

    const auto cond_eval = Teuchos::rcp(
        new BoundaryCondition::
            FullInductionConducting<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, bc_params, mhd_props));

    test_fixture.registerEvaluator<EvalType>(cond_eval);

    // Add required test fields.
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            cond_eval->_boundary_induced_magnetic_field[dim]);
        test_fixture.registerTestField<EvalType>(
            cond_eval->_boundary_grad_induced_magnetic_field[dim]);
    }
    if (build_magn_corr)
    {
        test_fixture.registerTestField<EvalType>(
            cond_eval->_boundary_scalar_magnetic_potential);
    }

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto bnd_magn_field_0_result
        = test_fixture.getTestFieldData<EvalType>(
            cond_eval->_boundary_induced_magnetic_field[0]);
    const auto bnd_grad_magn_field_0_result
        = test_fixture.getTestFieldData<EvalType>(
            cond_eval->_boundary_grad_induced_magnetic_field[0]);
    const auto bnd_magn_field_1_result
        = test_fixture.getTestFieldData<EvalType>(
            cond_eval->_boundary_induced_magnetic_field[1]);
    const auto bnd_grad_magn_field_1_result
        = test_fixture.getTestFieldData<EvalType>(
            cond_eval->_boundary_grad_induced_magnetic_field[1]);

    // Set up expected values based on case
    // For now VERTEX only supports num_grad_dim = num_space_dim,
    // but the reference script can produce values for the case
    // where num_grad_dim = 2 and num_space_dim = 3
    const Kokkos::Array<double, 3> exp_b_2d = {1.307375, 2.417125, nanval};
    const Kokkos::Array<double, 3> exp_b_3d = {1.2365, 2.5195, 3.7395};

    const auto exp_b = num_space_dim == 2 ? exp_b_2d : exp_b_3d;

    const Kokkos::Array<Kokkos::Array<double, 3>, 3> exp_grad_b_2d
        = {{{7.16375, 0.04125, nanval},
            {-3.9075, -0.0225, nanval},
            {nanval, nanval, nanval}}};
    const Kokkos::Array<Kokkos::Array<double, 3>, 3> exp_grad_b_2d_res
        = {{{7.26978125, -0.11190625, nanval},
            {-3.83409375, -0.12853125, nanval},
            {nanval, nanval, nanval}}};
    const Kokkos::Array<Kokkos::Array<double, 3>, 3> exp_grad_b_3d
        = {{{6.730625, 0.666875, -0.570625},
            {-3.67125, -0.36375, 0.31125},
            {1.98859375, 0.19703125, -0.16859375}}};
    const Kokkos::Array<Kokkos::Array<double, 3>, 3> exp_grad_b_3d_res
        = {{{6.81244062, 0.54869688, -0.50699062},
            {-3.4704, -0.65386667, 0.46746667},
            {2.25640937, -0.18981354, 0.03970729}}};

    const auto exp_grad_b = num_space_dim == 2     ? build_resistive_flux
                                                         ? exp_grad_b_2d_res
                                                         : exp_grad_b_2d
                            : build_resistive_flux ? exp_grad_b_3d_res
                                                   : exp_grad_b_3d;

    const double exp_scalar_magn_pot
        = dirichlet_scalar_magn_pot ? bnd_scalar_magn_pot : scalar_magn_pot;

    const auto& ir = *test_fixture.ir;
    const int num_point = ir.num_points;
    const double tol = build_resistive_flux ? 6.0e-9 : 1.0e-12;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_b[0], fieldValue(bnd_magn_field_0_result, 0, qp));
        EXPECT_DOUBLE_EQ(exp_b[1], fieldValue(bnd_magn_field_1_result, 0, qp));
        if (num_space_dim == 3)
        {
            const auto bnd_magn_field_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    cond_eval->_boundary_induced_magnetic_field[2]);
            EXPECT_DOUBLE_EQ(exp_b[2],
                             fieldValue(bnd_magn_field_2_result, 0, qp));
        }
        if (build_magn_corr)
        {
            const auto bnd_scalar_magn_pot_result
                = test_fixture.getTestFieldData<EvalType>(
                    cond_eval->_boundary_scalar_magnetic_potential);
            EXPECT_DOUBLE_EQ(exp_scalar_magn_pot,
                             fieldValue(bnd_scalar_magn_pot_result, 0, qp));
        }

        for (int d = 0; d < num_space_dim; ++d)
        {
            EXPECT_NEAR(exp_grad_b[0][d],
                        fieldValue(bnd_grad_magn_field_0_result, 0, qp, d),
                        tol);
            EXPECT_NEAR(exp_grad_b[1][d],
                        fieldValue(bnd_grad_magn_field_1_result, 0, qp, d),
                        tol);
            if (num_space_dim == 3)
            {
                const auto bnd_grad_magn_field_2_result
                    = test_fixture.getTestFieldData<EvalType>(
                        cond_eval->_boundary_grad_induced_magnetic_field[2]);
                EXPECT_NEAR(exp_grad_b[2][d],
                            fieldValue(bnd_grad_magn_field_2_result, 0, qp, d),
                            tol);
            }
        }
    }
}

//---------------------------------------------------------------------------//
struct FullInductionConductingTestParams
{
    bool build_magn_corr;
    bool dirichlet_scalar_magn_pot;
    bool build_resistive_flux;
};

class FullInductionConducting
    : public testing::TestWithParam<FullInductionConductingTestParams>
{
  public:
    struct PrintNameString
    {
        template<class T>
        std::string operator()(const testing::TestParamInfo<T>& info) const
        {
            auto p = static_cast<FullInductionConductingTestParams>(info.param);

            const std::string base_name = p.build_resistive_flux ? "Resistive"
                                                                 : "";
            const std::string pot_name = !p.build_magn_corr ? "NoCleaning"
                                         : p.dirichlet_scalar_magn_pot
                                             ? "FixedScalarMagneticPotential"
                                             : "FreeScalarMagneticPotential";
            return base_name + pot_name;
        }
    };
};

//---------------------------------------------------------------------------//
TEST_P(FullInductionConducting, Residual2d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Residual, 2>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot,
                                          params.build_resistive_flux);
}

//---------------------------------------------------------------------------//
TEST_P(FullInductionConducting, Jacobian2d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot,
                                          params.build_resistive_flux);
}

//---------------------------------------------------------------------------//
TEST_P(FullInductionConducting, Residual3d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Residual, 3>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot,
                                          params.build_resistive_flux);
}

//---------------------------------------------------------------------------//
TEST_P(FullInductionConducting, Jacobian3d)
{
    auto params = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(params.build_magn_corr,
                                          params.dirichlet_scalar_magn_pot,
                                          params.build_resistive_flux);
}

//---------------------------------------------------------------------------//
INSTANTIATE_TEST_SUITE_P(
    Test,
    FullInductionConducting,
    testing::Values(FullInductionConductingTestParams{false, false, false},
                    FullInductionConductingTestParams{false, false, true},
                    FullInductionConductingTestParams{true, false, false},
                    FullInductionConductingTestParams{true, false, true},
                    FullInductionConductingTestParams{true, true, false},
                    FullInductionConductingTestParams{true, true, true}),
    FullInductionConducting::PrintNameString());

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
