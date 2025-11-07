#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "VertexCFD_Utils_PhaseIndex.hpp"
#include "VertexCFD_Utils_ScalarToVector.hpp"
#include "VertexCFD_Utils_VelocityDim.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <string>
#include <vector>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<typename EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    const int num_scalar_dim_all;

    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>> foo_scalars;
    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>>
        dxdt_foo_scalars;
    std::vector<PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>>
        grad_foo_scalars;

    Dependencies(const panzer::IntegrationRule& ir,
                 const std::string& base_name,
                 const int num_scalar_dim)
        : num_scalar_dim_all{num_scalar_dim}
    {
        foo_scalars.reserve(num_scalar_dim);
        dxdt_foo_scalars.reserve(num_scalar_dim);
        grad_foo_scalars.reserve(num_scalar_dim);
        for (int dim = 0; dim < num_scalar_dim; ++dim)
        {
            // Omit a scalar to test optional dependent scalars.
            if (dim == 4)
                continue;

            const auto name = base_name + std::to_string(dim);

            foo_scalars.emplace_back(name, ir.dl_scalar);
            dxdt_foo_scalars.emplace_back("DXDT_" + name, ir.dl_scalar);
            grad_foo_scalars.emplace_back("GRAD_" + name, ir.dl_vector);

            this->addEvaluatedField(foo_scalars.back());
            this->addEvaluatedField(dxdt_foo_scalars.back());
            this->addEvaluatedField(grad_foo_scalars.back());
        }

        this->setName("ScalarToVector Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData workset) override
    {
        const int num_scalar_dim = foo_scalars.size();

        for (int scalar_dim = 0, foo_value = 1; scalar_dim < num_scalar_dim;
             ++scalar_dim)
        {
            // Create host mirrors of fields
            auto foo_view = foo_scalars[scalar_dim].get_view();
            auto dxdt_foo_view = dxdt_foo_scalars[scalar_dim].get_view();
            auto grad_foo_view = grad_foo_scalars[scalar_dim].get_view();
            auto host_foo = Kokkos::create_mirror(foo_view);
            auto host_dxdt_foo = Kokkos::create_mirror(dxdt_foo_view);
            auto host_grad_foo = Kokkos::create_mirror(grad_foo_view);

            if (scalar_dim == 4)
                foo_value += foo_view.size();

            for (int cell = 0; cell < workset.num_cells; ++cell)
            {
                const int num_point = foo_view.extent(1);
                for (int point = 0; point < num_point; ++point, ++foo_value)
                {
                    host_foo(cell, point) = foo_value;

                    host_dxdt_foo(cell, point) = 100 + foo_value;

                    const int num_grad_dim = grad_foo_view.extent(2);
                    for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
                    {
                        host_grad_foo(cell, point, grad_dim)
                            = 1000 + 100 * grad_dim + foo_value;
                    }
                }
            }
            Kokkos::deep_copy(foo_view, host_foo);
            Kokkos::deep_copy(dxdt_foo_view, host_dxdt_foo);
            Kokkos::deep_copy(grad_foo_view, host_grad_foo);
        }
    }
};

enum class ScalarNaming
{
    Indexed,
    Suffixed,
    List
};

template<typename EvalType, typename DimTag>
void testEval(const int num_scalar_dim,
              const int num_grad_dim,
              const ScalarNaming scalar_naming)
{
    // Setup test fixture.
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    std::vector<std::string> name_tags;
    std::string base_name = "foo_";
    if (scalar_naming == ScalarNaming::Suffixed)
    {
        base_name += "bar_";
        name_tags.reserve(num_scalar_dim);
        for (int i = 0; i < num_scalar_dim; ++i)
            name_tags.emplace_back("bar_" + std::to_string(i));
    }
    else if (scalar_naming == ScalarNaming::List)
    {
        base_name += "bar_";
        name_tags.reserve(num_scalar_dim);
        for (int i = 0; i < num_scalar_dim; ++i)
        {
            if (i == 4)
                continue;
            name_tags.emplace_back(base_name + std::to_string(i));
        }
    }

    // Create test dependency to set initial fields
    auto deps = Teuchos::rcp(
        new Dependencies<EvalType>(ir, base_name, num_scalar_dim));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Create evaluator.
    auto eval = [&]() {
        if (name_tags.empty())
        {
            return Utils::ScalarToVector<EvalType, DimTag>::createFromIndexed(
                ir, "foo", num_scalar_dim, true, true);
        }
        else if (scalar_naming == ScalarNaming::Suffixed)
        {
            std::vector<std::optional<std::string>> opt_name_tags(
                name_tags.begin(), name_tags.end());
            // Omit a scalar to test optional dependent scalars.
            if (num_scalar_dim > 4)
                opt_name_tags[4].reset();
            return Utils::ScalarToVector<EvalType, DimTag>::createFromSuffixed(
                ir, "foo", opt_name_tags, true, true);
        }
        else
        {
            // Omit a scalar to test optional dependent scalars.
            // if (num_scalar_dim > 4)
            //    name_tags.erase(name_tags.begin() + 4);
            return Utils::ScalarToVector<EvalType, DimTag>::createFromList(
                ir, "foo", name_tags, true, true);
        }
    }();
    test_fixture.registerEvaluator<EvalType>(eval);

    EXPECT_EQ("ScalarToVector (foo)", eval->getName());

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(eval->_vector_field);
    test_fixture.registerTestField<EvalType>(eval->_vector_dxdt_field);
    test_fixture.registerTestField<EvalType>(eval->_vector_grad_field);

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto vector_foo
        = test_fixture.getTestFieldData<EvalType>(eval->_vector_field);
    const auto vector_dxdt_foo
        = test_fixture.getTestFieldData<EvalType>(eval->_vector_dxdt_field);
    const auto vector_grad_foo
        = test_fixture.getTestFieldData<EvalType>(eval->_vector_grad_field);

    // Check the solution
    const int num_point = ir.num_points;
    for (int scalar_dim = 0; scalar_dim < num_scalar_dim; ++scalar_dim)
    {
        for (int qp = 0; qp < num_point; ++qp)
        {
            double expected = 0.0;

            // Test foo
            if (scalar_dim != 4)
            {
                expected = num_point * scalar_dim + qp + 1;
                EXPECT_EQ(expected, fieldValue(vector_foo, 0, qp, scalar_dim))
                    << "  foo(" << 0 << ", " << qp << ", " << scalar_dim
                    << ')';
            }
            else
            {
                EXPECT_TRUE(
                    std::isnan(fieldValue(vector_foo, 0, qp, scalar_dim)))
                    << "  foo(" << 0 << ", " << qp << ", " << scalar_dim
                    << ')';
            }

            // Test dxdt_foo
            if (scalar_dim != 4)
            {
                expected = 100 + num_point * scalar_dim + qp + 1;
                EXPECT_EQ(expected,
                          fieldValue(vector_dxdt_foo, 0, qp, scalar_dim))
                    << "  dxdt_foo(" << 0 << ", " << qp << ", " << scalar_dim
                    << ')';
            }
            else
            {
                EXPECT_TRUE(
                    std::isnan(fieldValue(vector_dxdt_foo, 0, qp, scalar_dim)))
                    << "  dxdt_foo(" << 0 << ", " << qp << ", " << scalar_dim
                    << ')';
            }

            // Test grad_foo
            for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
            {
                if (scalar_dim != 4)
                {
                    expected = 1000 + 100 * grad_dim + num_point * scalar_dim
                               + qp + 1;
                    EXPECT_EQ(expected,
                              fieldValue(
                                  vector_grad_foo, 0, qp, grad_dim, scalar_dim))
                        << "  grad_foo(" << 0 << ", " << qp << ", " << grad_dim
                        << ", " << scalar_dim << ')';
                }
                else
                {
                    EXPECT_TRUE(std::isnan(fieldValue(
                        vector_grad_foo, 0, qp, grad_dim, scalar_dim)))
                        << "  grad_foo(" << 0 << ", " << qp << ", " << grad_dim
                        << ", " << scalar_dim << ')';
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
// Test residual and Jacobian evaluations.
template<typename EvalType>
class ScalarToVectorTest : public ::testing::Test
{
};
using EvalTypes
    = ::testing::Types<panzer::Traits::Residual, panzer::Traits::Jacobian>;

class EvalTypeNames
{
  public:
    template<typename T>
    static std::string GetName(const int i)
    {
        if (std::is_same<T, panzer::Traits::Residual>())
            return "Residual";
        if (std::is_same<T, panzer::Traits::Jacobian>())
            return "Jacobian";
        return std::to_string(i);
    }
};

TYPED_TEST_SUITE(ScalarToVectorTest, EvalTypes, EvalTypeNames);

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Velocity22)
{
    testEval<TypeParam, VelocityDim>(2, 2, ScalarNaming::Indexed);
}

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Velocity33)
{
    testEval<TypeParam, VelocityDim>(3, 3, ScalarNaming::Indexed);
}

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Velocity32)
{
    testEval<TypeParam, VelocityDim>(3, 2, ScalarNaming::Indexed);
}

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Phase52Suffixed)
{
    testEval<TypeParam, PhaseIndex>(5, 2, ScalarNaming::Suffixed);
}

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Phase72Suffixed)
{
    testEval<TypeParam, PhaseIndex>(7, 2, ScalarNaming::Suffixed);
}

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Phase22List)
{
    testEval<TypeParam, PhaseIndex>(2, 2, ScalarNaming::List);
}

//---------------------------------------------------------------------------//
TYPED_TEST(ScalarToVectorTest, Phase32List)
{
    testEval<TypeParam, PhaseIndex>(3, 2, ScalarNaming::List);
}

} // namespace Test
} // namespace VertexCFD
