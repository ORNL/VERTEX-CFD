#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "observers/VertexCFD_TempusObserver_WriteMatrix.hpp"

#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_MultiVectorIn.h>
#include <Epetra_MpiComm.h>
#include <NOX_Utils.H>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Tempus_IntegratorBasic.hpp>
#include <Tempus_StepperForwardEuler.hpp>
#include <Teuchos_RCP.hpp>
#include <Trilinos_version.h>

#include <CDR_Model.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
class WriteMatrixTester : public ::testing::Test
{
  public:
    void SetUp()
    {
        // Create an Epetra communicator
        epetra_comm_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

        // Set time steps where matrices will be written
        write_steps_.resize(3);
        write_steps_[0] = 1;
        write_steps_[1] = 5;
        write_steps_[2] = 19;
    }

    void TearDown() {}

    void solve() { integrator_->advanceTime(); }

    void checkResults() const
    {
        for (int val : write_steps_)
        {
            // Test Jacobian matrix
            std::string filename = "jacobian-";
            filename += std::to_string(val) + ".mtx";

            Epetra_CrsMatrix* A;
            bool transpose = false;
            int ierr = EpetraExt::MatrixMarketFileToCrsMatrix(
                filename.c_str(), *epetra_comm_, A, transpose);

            EXPECT_EQ(0, ierr);
            int num_global_rows = 101;
            EXPECT_EQ(num_global_rows, A->NumGlobalRows());
            EXPECT_EQ(num_global_rows, A->NumGlobalCols());
            EXPECT_EQ(301, A->NumGlobalNonzeros());

            // Check matrix values. Except for boundaries, the stencil is
            //  essentially [-100 200 100] to within a couple of digits.
            // The problem is essentially linear
            //  and static, so all matrices will be nearly the same.
            int* col_indices;
            double* values;
            int num_entries;
            int num_local_rows = A->NumMyRows();
            int global_row;
            for (int local_row = 0; local_row < num_local_rows; ++local_row)
            {
                A->ExtractMyRowView(
                    local_row, num_entries, values, col_indices);
                global_row = A->GRID(local_row);
                for (int entry = 0; entry < num_entries; ++entry)
                {
                    int global_col = A->GCID(col_indices[entry]);
                    if (global_row == 0)
                    {
                        EXPECT_EQ(2, num_entries);
                        if (global_col == 0)
                        {
                            // Entry (0, 0)
                            EXPECT_NEAR(1.0, values[entry], 1e-6);
                        }
                        else
                        {
                            // Entry (0, 1)
                            EXPECT_DOUBLE_EQ(0.0, values[entry]);
                        }
                    }
                    else if (global_row == (num_global_rows - 1))
                    {
                        EXPECT_EQ(2, num_entries);
                        if (global_col == (num_global_rows - 1))
                        {
                            // Diagonal entry in last row
                            EXPECT_NEAR(99.5, values[entry], 0.1);
                        }
                        else
                        {
                            // Off-diagonal entry in last row
                            EXPECT_NEAR(-99.5, values[entry], 0.1);
                        }
                    }
                    else
                    {
                        EXPECT_EQ(3, num_entries);
                        if (global_col == global_row)
                        {
                            // Diagonal entry
                            EXPECT_NEAR(198., values[entry], 0.1);
                        }
                        else if (global_col == (global_row - 1))
                        {
                            // Left of diagonal
                            EXPECT_NEAR(-99.5, values[entry], 0.1);
                        }
                        else if (global_col == (global_row + 1))
                        {
                            // Right of diagonal entry
                            EXPECT_NEAR(-98.5, values[entry], 0.1);
                        }
                    }
                }
            }

            // Test Residual
            filename = "residual-";
            filename += std::to_string(val) + ".mtx";
            Epetra_MultiVector* resid;
            int err = EpetraExt::MatrixMarketFileToMultiVector(
                filename.c_str(), A->RowMap(), resid);
            EXPECT_EQ(0, err);
            int global_length = 101;
            EXPECT_EQ(global_length, resid->GlobalLength());
            std::vector<double> minval(1), maxval(1), meanval(1);
            resid->MinValue(minval.data());
            resid->MaxValue(maxval.data());
            resid->MeanValue(meanval.data());
            if (val == 1)
            {
                EXPECT_NEAR(-3.76e-8, minval[0], 1e-10);
                EXPECT_NEAR(1.48e-8, maxval[0], 1e-10);
                EXPECT_NEAR(1.49e-9, meanval[0], 1e-11);
            }
            else if (val == 5)
            {
                EXPECT_NEAR(-4.66e-8, minval[0], 1e-10);
                EXPECT_NEAR(1.86e-8, maxval[0], 1e-10);
                EXPECT_NEAR(3.94e-9, meanval[0], 1e-11);
            }
            else if (val == 19)
            {
                EXPECT_NEAR(-7.04e-14, minval[0], 1e-15);
                EXPECT_EQ(0, maxval[0]);
                EXPECT_NEAR(-3.29e-14, meanval[0], 1e-16);
            }

            // We must call delete on EpetraExt-allocated object
            delete A;
            delete resid;
        }
    }

  protected:
    void buildEvaluator()
    {
        // Set parameters for convection-diffusion-reaction model
        int num_elements = 100;
        double zmin = 0.0;
        double zmax = 1.0;
        double a = 1.0; // convection coefficient
        double k = 2.0; // source
        auto cdr_model = Teuchos::rcp(new CDR_Model<double>(
            epetra_comm_, num_elements, zmin, zmax, a, k));

        // CDR model doesn't build its own linear solver, need to supply
        // factory
        Stratimikos::DefaultLinearSolverBuilder builder;

        auto stratimikos_params = Teuchos::parameterList();
        stratimikos_params->set("Linear Solver Type", "Belos");
        stratimikos_params->set("Preconditioner Type", "None");
        builder.setParameterList(stratimikos_params);

        Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<double>> lows_factory
            = builder.createLinearSolveStrategy("");

        cdr_model->set_W_factory(lows_factory);
        model_ = cdr_model;
    }

    void buildIntegrator()
    {
        // Set Tempus parameters
        auto tempus_params = Teuchos::parameterList("Tempus");
        tempus_params->set("Integrator Name", "Default Integrator");

        auto integrator_params
            = Teuchos::sublist(tempus_params, "Default Integrator");
        integrator_params->set("Integrator Type", "Integrator Basic");
        integrator_params->set("Stepper Name", "Default Stepper");

        auto control_params
            = Teuchos::sublist(integrator_params, "Time Step Control");
        control_params->set("Initial Time", 0.);
        control_params->set("Final Time", 20.);
        control_params->set("Initial Time Step", 1.);
        control_params->set("Minimum Time Step", 1.);
        control_params->set("Maximum Time Step", 1.);

        auto stepper_params
            = Teuchos::sublist(tempus_params, "Default Stepper");
        stepper_params->set("Stepper Type", "Backward Euler");

        // Linear solver
        stepper_params->set("Solver Name", "Default Solver");
        auto solver_params = Teuchos::sublist(stepper_params, "Default Solver");
        auto nox_params = Teuchos::sublist(solver_params, "NOX");
        nox_params->set(
            "Output Information",
            NOX::Utils::StepperIteration + NOX::Utils::StepperDetails);

        // Setup time integrator.
#if TRILINOS_MAJOR_MINOR_VERSION >= 130100
        integrator_
            = Tempus::createIntegratorBasic<double>(tempus_params, model_);
#else
        integrator_ = Tempus::integratorBasic<double>(tempus_params, model_);
#endif

        // Build observer to write matrix to file
        auto write_params = Teuchos::parameterList();
        write_params->set("Write Matrix", true);
        write_params->set("Write Residual", true);
        write_params->set("Write Steps", write_steps_);
        auto matrix_writer = Teuchos::rcp(
            new VertexCFD::TempusObserver::WriteMatrix<double>(*write_params));
        integrator_->setObserver(matrix_writer);

        // Initialize integrator
        integrator_->initialize();
    }

    Teuchos::RCP<Epetra_Comm> epetra_comm_;
    Teuchos::RCP<Thyra::ModelEvaluator<double>> model_;
    Teuchos::RCP<Tempus::IntegratorBasic<double>> integrator_;
    Teuchos::Array<int> write_steps_;
};

//---------------------------------------------------------------------------//
TEST_F(WriteMatrixTester, cdr_test)
{
    // This test only has 100 degrees of freedom in the system and bogs down
    // if run on a large number of processors.
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs > 16)
        return;

    this->buildEvaluator();
    this->buildIntegrator();
    this->solve();
    this->checkResults();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
