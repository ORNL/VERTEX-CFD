#include "VertexCFD_LinearSolverUnitTestConfig.hpp"

#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <VertexCFD_SolverTester.hpp>
#include <linear_solvers/VertexCFD_LinearSolvers_HyprePreconditionerFactory.hpp>

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_NodeType.hpp>
#include <Thyra_DefaultLinearOpSource.hpp>
#include <Thyra_TpetraLinearOp.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
class PreconditionerFactoryTester : public SolverTester
{
};

//---------------------------------------------------------------------------//
TEST_F(PreconditionerFactoryTester, build_test)
{
    using panzer::GlobalOrdinal;
    using panzer::TpetraNodeType;

    // Wrap matrix into Thyra operator
    auto thyra_op
        = Thyra::createLinearOp<double, int, GlobalOrdinal, TpetraNodeType>(
            _matrix);
    auto thyra_op_src
        = Teuchos::rcp(new Thyra::DefaultLinearOpSource<double>(thyra_op));

    // Create factory
    VertexCFD::LinearSolvers::HyprePreconditionerFactory factory;
    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    std::string yaml_dir = VERTEXCFD_LINEARSOLVER_TEST_DATA_DIR;
    std::string yaml_file = "hypre.yml";
    std::string yaml_location = yaml_dir + yaml_file;
    params->set("HypreDrive YAML File", yaml_location);
    factory.setParameterList(params);
    EXPECT_TRUE(factory.isCompatible(*thyra_op_src));

    // Create preconditioner and initialize
    auto prec = factory.createPrec();
    EXPECT_TRUE(prec != Teuchos::null);
    factory.initializePrec(thyra_op_src, prec.get());

    // Check that operator was initialized correctly within preconditioner
    auto unspecified_prec_op = prec->getUnspecifiedPrecOp();
    EXPECT_TRUE(unspecified_prec_op != Teuchos::null);

    // Extract Thyra operator
    auto thyra_tpetra_prec = Teuchos::rcp_dynamic_cast<
        const Thyra::TpetraLinearOp<double, int, GlobalOrdinal, TpetraNodeType>>(
        unspecified_prec_op);
    EXPECT_TRUE(thyra_tpetra_prec != Teuchos::null);
    auto tpetra_prec = thyra_tpetra_prec->getConstTpetraOperator();
    EXPECT_TRUE(tpetra_prec != Teuchos::null);

    // Check if operator is an Ifpack preconditioner.
    auto ifpack_prec
        = Teuchos::rcp_dynamic_cast<const Ifpack2::Preconditioner<>>(
            tpetra_prec);
    EXPECT_TRUE(ifpack_prec != Teuchos::null);
    EXPECT_TRUE(ifpack_prec->isInitialized());
    EXPECT_TRUE(ifpack_prec->isComputed());

    // Perform operator apply
    ifpack_prec->apply(*_x, *_y, Teuchos::NO_TRANS, 1.0, 0.0);

    // Compare against reference solution
    auto y_data = _y->getData(0);
    int num_local_rows = y_data.size();
    double tol = 1e-14;
    for (int local_row = 0; local_row < num_local_rows; ++local_row)
    {
        EXPECT_NEAR(_ref_soln[local_row], y_data[local_row], tol);
    }
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
