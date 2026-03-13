#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTauSUPG.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
// Tau model for NS equations
enum class TauModel
{
    Steady,
    Transient,
    NoSUPG,
};

// Tau model for temperature equation
enum class TempTauModel
{
    TempMultiDSUPGSteady,
    TempMultiDSUPGTransient,
    TempNoSUPG,
    TempOneDXNodal,
    TempOneDXUpwind
};

//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> ele_lngth;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> cp;

    TempTauModel _temp_tau_model;
    bool _solve_temp;

    Dependencies(const panzer::IntegrationRule& ir,
                 const TempTauModel temp_tau_model,
                 const bool solve_temp)
        : velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , ele_lngth("element_length", ir.dl_vector)
        , rho("density", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , k("thermal_conductivity", ir.dl_scalar)
        , cp("specific_heat_capacity", ir.dl_scalar)
        , _temp_tau_model(temp_tau_model)
        , _solve_temp(solve_temp)
    {
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(ele_lngth);
        this->addEvaluatedField(nu);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(k);
        this->addEvaluatedField(cp);

        this->setName("Incompressible Tau SUPG Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "tau supg test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = ele_lngth.extent(1);
        const int num_space_dim = ele_lngth.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            velocity_0(c, qp) = 0.5;
            velocity_1(c, qp) = 1.5;
            velocity_2(c, qp) = num_space_dim == 3 ? 2.5 : _nanval;
            rho(c, qp) = _temp_tau_model == TempTauModel::TempMultiDSUPGSteady
                             ? 3.0
                             : 1.0;
            nu(c, qp) = 0.1;
            k(c, qp) = _solve_temp ? 2.0 : _nanval;
            cp(c, qp) = _solve_temp ? 10.0 : _nanval;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                ele_lngth(c, qp, dim) = 3.0 * (dim + 1);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const TempTauModel temp_tau_model,
              const TauModel mom_tau_model,
              const bool solve_temp)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(
        new Dependencies<EvalType>(ir, temp_tau_model, solve_temp));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList fluid_params;
    fluid_params.set("Build Temperature Equation", solve_temp);

    double cont_tau_supg_exp;
    double mom_tau_supg_exp;
    double temp_tau_supg_exp;
    Teuchos::ParameterList closure_params;

    // Navier-Stokes equations
    if (mom_tau_model == TauModel::Steady)
    {
        closure_params.set("Tau model for Navier-Stokes equations", "Steady");
        closure_params.set("SUPG coefficient for momentum equations", 2.0);
        cont_tau_supg_exp = 0.6109184438110759;
        mom_tau_supg_exp = 2.4436737752443034;
    }
    else if (mom_tau_model == TauModel::Transient)
    {
        test_fixture.setStepSize(0.2);
        closure_params.set("SUPG coefficient for continuity equation", 3.0);
        closure_params.set("Tau model for Navier-Stokes equations",
                           "Transient");
        cont_tau_supg_exp = 0.29945873503727904;
        mom_tau_supg_exp = 0.04990978917287984;
    }
    else
    {
        closure_params.set("Tau model for Navier-Stokes equations", "NoSUPG");
        mom_tau_supg_exp = 0.0;
        cont_tau_supg_exp = 0.0;
    }

    // Temperature equation
    if (solve_temp)
    {
        if (temp_tau_model == TempTauModel::TempOneDXNodal)
        {
            closure_params.set("Tau model for temperature equation",
                               "TempOneDXNodal");
            closure_params.set("SUPG coefficient for temperature equation",
                               2.0);
            temp_tau_supg_exp = 8.81328119433642;
        }
        else if (temp_tau_model == TempTauModel::TempOneDXUpwind)
        {
            closure_params.set("Tau model for temperature equation",
                               "TempOneDXUpwind");
            temp_tau_supg_exp = 2.999999940000001;
        }
        else if (temp_tau_model == TempTauModel::TempMultiDSUPGSteady)
        {
            closure_params.set("Tau model for temperature equation",
                               "TempMultiDSUPGSteady");
            temp_tau_supg_exp = 0.6109414239741788;
        }
        else if (temp_tau_model == TempTauModel::TempMultiDSUPGTransient)
        {
            test_fixture.setStepSize(0.2);
            closure_params.set("Tau model for temperature equation",
                               "TempMultiDSUPGTransient");
            temp_tau_supg_exp = 0.04990925872954427;
        }
        else
        {
            closure_params.set("Tau model for temperature equation",
                               "TempNoSUPG");
            temp_tau_supg_exp = 0.0;
        }
    }

    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleTauSUPG<EvalType, panzer::Traits, num_space_dim>(
                ir, fluid_params, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_tau_supg_cont);
    test_fixture.registerTestField<EvalType>(eval->_tau_supg_mom);
    if (solve_temp)
        test_fixture.registerTestField<EvalType>(eval->_tau_supg_energy);
    test_fixture.evaluate<EvalType>();

    const auto cont_tau_supg
        = test_fixture.getTestFieldData<EvalType>(eval->_tau_supg_cont);
    const auto mom_tau_supg
        = test_fixture.getTestFieldData<EvalType>(eval->_tau_supg_mom);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(cont_tau_supg_exp, fieldValue(cont_tau_supg, 0, qp));
        EXPECT_DOUBLE_EQ(mom_tau_supg_exp, fieldValue(mom_tau_supg, 0, qp));
        if (solve_temp)
        {
            const auto temp_tau_supg = test_fixture.getTestFieldData<EvalType>(
                eval->_tau_supg_energy);
            EXPECT_DOUBLE_EQ(temp_tau_supg_exp,
                             fieldValue(temp_tau_supg, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGNSTransient2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        TempTauModel::TempOneDXNodal, TauModel::Transient, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGNSTransient2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        TempTauModel::TempOneDXNodal, TauModel::Transient, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGNodal2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        TempTauModel::TempOneDXNodal, TauModel::NoSUPG, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGNodal2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        TempTauModel::TempOneDXNodal, TauModel::NoSUPG, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGUpwind2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        TempTauModel::TempOneDXUpwind, TauModel::Transient, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGUpwind2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        TempTauModel::TempOneDXUpwind, TauModel::Transient, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGMuliDSteady3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        TempTauModel::TempMultiDSUPGSteady, TauModel::Steady, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGMultiSteady3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        TempTauModel::TempMultiDSUPGSteady, TauModel::Steady, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGMuliDTransient2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        TempTauModel::TempMultiDSUPGTransient, TauModel::Transient, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGMultiTransient2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        TempTauModel::TempMultiDSUPGTransient, TauModel::Transient, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGNoSUPG2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        TempTauModel::TempNoSUPG, TauModel::NoSUPG, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleTauSUPGNoSUPG2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        TempTauModel::TempNoSUPG, TauModel::NoSUPG, true);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.model_params.set("Tau model for Navier-Stokes equations",
                                  "Transient");
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleTauSUPG";
    test_fixture.eval_name = "Incompressible Tau SUPG "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleTauSUPG<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleTauSUPG_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleTauSUPG_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleTauSUPG_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleTauSUPG_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
