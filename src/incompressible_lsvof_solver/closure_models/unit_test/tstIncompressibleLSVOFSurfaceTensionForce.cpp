#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFSurfaceTensionForce.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_phase_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_phase_1;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_phase_0("GRAD_pha0", ir.dl_vector)
        , grad_phase_1("GRAD_pha1", ir.dl_vector)
    {
        this->addEvaluatedField(grad_phase_0);
        this->addEvaluatedField(grad_phase_1);

        this->setName(
            "Incompressible LSVOF Surface Tension force for Momentum "
            "Equation Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF surface tension force test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_phase_0.extent(1);
        const int num_space_dim = grad_phase_0.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_phase_0(c, qp, dim) = (0.25 + 0.5) * dimqp;
                grad_phase_1(c, qp, dim) = 0.25 * dimqp;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Surface Tension Coefficient", 0.7);
    closure_params.set<int>("Number of Phases", 3);

    // Phase names
    std::vector<std::string> phase_list;
    phase_list.push_back("pha0");
    phase_list.push_back("pha1");
    phase_list.push_back("pha2");

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFSurfaceTensionForce<EvalType,
                                                                 panzer::Traits,
                                                                 num_space_dim>(
            ir, closure_params, phase_list));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(
            eval->_surface_tension_flux[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_surftension_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_surface_tension_flux[0]);
    const auto fc_surftension_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_surface_tension_flux[1]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_surftension_0_flux_3d[3]
        = {2.432077301403062, 0.37416573867739417, -0.5612486080160912};
    const double exp_surftension_1_flux_3d[3]
        = {0.37416573867739417, 1.8708286933869702, 1.1224972160321824};
    const double exp_surftension_2_flux_3d[3]
        = {-0.5612486080160912, 1.1224972160321824, 0.9354143466934851};

    const double exp_surftension_0_flux_2d[2]
        = {1.2521980673998823, 0.626099033699941};
    const double exp_surftension_1_flux_2d[2]
        = {0.626099033699941, 0.3130495168499706};

    const double* exp_surftension_0_flux = num_space_dim == 3
                                               ? exp_surftension_0_flux_3d
                                               : exp_surftension_0_flux_2d;
    const double* exp_surftension_1_flux = num_space_dim == 3
                                               ? exp_surftension_1_flux_3d
                                               : exp_surftension_1_flux_2d;
    const double* exp_surftension_2_flux = exp_surftension_2_flux_3d;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_surftension_0_flux[dim],
                             fieldValue(fc_surftension_0, 0, qp, dim));
            EXPECT_DOUBLE_EQ(exp_surftension_1_flux[dim],
                             fieldValue(fc_surftension_1, 0, qp, dim));

            if (num_space_dim > 2) // 3D mesh
            {
                const auto fc_surftension_2
                    = test_fixture.getTestFieldData<EvalType>(
                        eval->_surface_tension_flux[2]);
                EXPECT_DOUBLE_EQ(exp_surftension_2_flux[dim],
                                 fieldValue(fc_surftension_2, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFSurfaceTensionForce2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFSurfaceTensionForce2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFSurfaceTensionForce3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFSurfaceTensionForce3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
