#include "VertexCFD_Utils_TypeTraits.hpp"

#include <Sacado.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
// Test ExpectedType against ResultType.
template<typename ExpectedResult, typename... Types>
void testResultType(const Types&...)
{
    using Result = ResultType<Types...>;
    ::testing::StaticAssertTypeEq<ExpectedResult, Result>();
}

//---------------------------------------------------------------------------//
// Test with a single argument of different types.
TEST(TypeTraits, OneArgument)
{
    // int
    testResultType<int>(1);

    // double
    testResultType<double>(1.0);

    // Static FAD
    using sfad_type = Sacado::Fad::SFad<double, 1>;
    const sfad_type a = 1.0;

    // SFad
    testResultType<sfad_type>(a);

    // SFad expr
    testResultType<sfad_type>(a * a + 1);

    // Dynamic FAD
    using dfad_type = Sacado::Fad::DFad<double>;
    const dfad_type x(1, 1.0);

    // DFad
    testResultType<dfad_type>(x);

    // DFad expr
    testResultType<dfad_type>(x * x + 1);

    // Nested  FAD -> DFad<SFad>
    using nfad_type = Sacado::Fad::DFad<sfad_type>;
    const nfad_type p(1, 1.0);

    // DFad<SFad>
    testResultType<nfad_type>(p);

    // DFad<SFad> expr
    testResultType<nfad_type>(p * p + 1);

    // DFad<SFad>/DFad/SFad expr
    testResultType<nfad_type>(a * x + p);
}

//---------------------------------------------------------------------------//
// Test with two arguments of various type combinations.
TEST(TypeTraits, TwoArgument)
{
    // int / int
    testResultType<int>(1, 2);

    // double / int
    testResultType<double>(1.0, 2);
    testResultType<double>(1, 2.0);

    // double / double
    testResultType<double>(1.0, 2.0);

    // Static FAD
    using sfad_type = Sacado::Fad::SFad<double, 1>;
    const sfad_type a = 1.0;
    const sfad_type b = 2.0;

    // SFad / int
    testResultType<sfad_type>(a, 1);
    testResultType<sfad_type>(1, a);

    // SFad / double
    testResultType<sfad_type>(a, 1.0);
    testResultType<sfad_type>(1.0, a);

    // SFad / SFad
    testResultType<sfad_type>(a, b);

    // SFad expr / int
    testResultType<sfad_type>(a * a + 1, 1);
    testResultType<sfad_type>(1, a * a + 1);

    // SFad expr / double
    testResultType<sfad_type>(a * a + 1, 1.0);
    testResultType<sfad_type>(1.0, a * a + 1);

    // SFad expr / SFad
    testResultType<sfad_type>(a * a + 1, b);
    testResultType<sfad_type>(b, a * a + 1);

    // SFad expr / SFad expr
    testResultType<sfad_type>(a * a + 1, 2 * b + b);

    // Dynamic FAD
    using dfad_type = Sacado::Fad::DFad<double>;
    const dfad_type x(1, 1.0);
    const dfad_type y(1, 1.0);

    // DFad / int
    testResultType<dfad_type>(x, 1);
    testResultType<dfad_type>(1, x);

    // DFad / double
    testResultType<dfad_type>(x, 1.0);
    testResultType<dfad_type>(1.0, x);

    // DFad / DFad
    testResultType<dfad_type>(x, y);

    // DFad expr / int
    testResultType<dfad_type>(x * x + 1, 1);
    testResultType<dfad_type>(1, x * x + 1);

    // DFad expr / double
    testResultType<dfad_type>(x * x + 1, 1.0);
    testResultType<dfad_type>(1.0, x * x + 1);

    // DFad expr / DFad
    testResultType<dfad_type>(x * x + 1, y);
    testResultType<dfad_type>(y, x * x + 1);

    // DFad expr / DFad expr
    testResultType<dfad_type>(x * x + 1, 2 * y + y);

    // Nested  FAD -> DFad<SFad>
    using nfad_type = Sacado::Fad::DFad<sfad_type>;
    const nfad_type p(1, 1.0);
    const nfad_type q(1, 2.0);

    // DFad<SFad> / int
    testResultType<nfad_type>(p, 1);
    testResultType<nfad_type>(1, p);

    // DFad<SFad> / double
    testResultType<nfad_type>(p, 1.0);
    testResultType<nfad_type>(1.0, p);

    // DFad<SFad> / SFad
    testResultType<nfad_type>(p, a);
    testResultType<nfad_type>(a, p);

    // DFad<SFad> / SFad expr
    testResultType<nfad_type>(p, a * a + 1);
    testResultType<nfad_type>(a * a + 1, p);

    // DFad<SFad> / DFad
    testResultType<nfad_type>(p, x);
    testResultType<nfad_type>(x, p);

    // DFad<SFad> / DFad expr
    testResultType<nfad_type>(p, x * x + 1);
    testResultType<nfad_type>(x * x + 1, p);

    // DFad<SFad> / DFad<SFad>
    testResultType<nfad_type>(p, q);

    // DFad<SFad> expr / int
    testResultType<nfad_type>(p * p + 1, 1);
    testResultType<nfad_type>(1, p * p + 1);

    // DFad<SFad> expr / double
    testResultType<nfad_type>(p * p + 1, 1.0);
    testResultType<nfad_type>(1.0, p * p + 1);

    // DFad<SFad> expr / SFad
    testResultType<nfad_type>(p * p + 1, a);
    testResultType<nfad_type>(a, p * p + 1);

    // DFad<SFad> expr / SFad expr
    testResultType<nfad_type>(p * p + 1, 2 * a + 1);
    testResultType<nfad_type>(2 * a + 1, p * p + 1);

    // DFad<SFad> expr / DFad
    testResultType<nfad_type>(p * p + 1, x);
    testResultType<nfad_type>(x, p * p + 1);

    // DFad<SFad> expr / DFad expr
    testResultType<nfad_type>(p * p + 1, 2 * x + 1);
    testResultType<nfad_type>(2 * x + 1, p * p + 1);

    // DFad<SFad> expr / DFad<SFad>
    testResultType<nfad_type>(p * p + 1, q);
    testResultType<nfad_type>(q, p * p + 1);

    // DFad<SFad> expr / DFad<SFad> expr
    testResultType<nfad_type>(p * p + 1, 2 * q + q);
}

//---------------------------------------------------------------------------//
// Test with many arguments of various types.
TEST(TypeTraits, ManyArgument)
{
    // int
    testResultType<int>(1, 2, 3, 4, 5);

    // double / int
    testResultType<double>(1, 2.0, 3, 4.0, 5);

    // Static FAD
    using sfad_type = Sacado::Fad::SFad<double, 1>;
    const sfad_type a = 1.0;
    const sfad_type b = 2.0;

    // SFad expr / SFad / double / int
    testResultType<sfad_type>(1, a, 2.0, a * b + 1, b);

    // Dynamic FAD
    using dfad_type = Sacado::Fad::DFad<double>;
    const dfad_type x(1, 1.0);
    const dfad_type y(1, 1.0);

    // DFad expr / DFad / double / int
    testResultType<dfad_type>(1, x, 2.0, x * y + 1, y);

    // Nested  FAD -> DFad<SFad>
    using nfad_type = Sacado::Fad::DFad<sfad_type>;
    const nfad_type p(1, 1.0);
    const nfad_type q(1, 2.0);

    // Everything
    testResultType<nfad_type>(
        1, p, 2.0, p * q + 1, 2 * x + y, x, a, a * b - 1, a + x + p);
}

//---------------------------------------------------------------------------//
} // namespace Test
