// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "VertexCFD_LinearSolvers_Hypre.hpp"

#include <BelosMultiVecTraits.hpp>
#include <BelosTpetraAdapter.hpp>

#include <Tpetra_Import.hpp>

#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <HYPRE_utilities.h>

#include <stdexcept>

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::rcpFromRef;

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Error code checking
void HYPRE_CHK_ERR(int code)
{
    if (code != 0)
    {
        char error_buffer[128];
        HYPRE_DescribeError(code, error_buffer);
        std::ostringstream ofs;
        ofs << "Hypre error with code" << code << ": " << error_buffer
            << std::endl;
        throw std::runtime_error(ofs.str());
    }
}

//---------------------------------------------------------------------------//
Hypre::Hypre()
    : is_initialized_(false)
    , is_computed_(false)
    , num_initialize_(0)
    , num_compute_(0)
    , num_apply_(0)
    , initialize_time_(0.0)
    , compute_time_(0.0)
    , apply_time_(0.0)
{
}

//==============================================================================
Hypre::~Hypre()
{
    destroy();
}

//==============================================================================
void Hypre::destroy()
{
    if (isInitialized())
    {
        HYPRE_CHK_ERR(HYPRE_IJMatrixDestroy(hypre_A_));
        HYPRE_CHK_ERR(HYPRE_IJVectorDestroy(hypre_x_));
        HYPRE_CHK_ERR(HYPRE_IJVectorDestroy(hypre_y_));
        HYPRE_CHK_ERR(HYPREDRV_PreconDestroy(hypredrv_));
        HYPRE_CHK_ERR(HYPREDRV_Destroy(&hypredrv_));
        HYPRE_CHK_ERR(HYPREDRV_Finalize());
    }
}

//==============================================================================
void Hypre::initialize()
{
    const std::string timer_name("LinearSolvers::Hypre::initialize");
    Teuchos::RCP<Teuchos::Time> timer
        = Teuchos::TimeMonitor::getNewTimer(timer_name);

    // Do nothing if already initialized.
    if (is_initialized_)
    {
        return;
    }

    // Scope initialize for timing.
    double startTime = timer->wallTime();
    {
        Teuchos::TimeMonitor timeMon(*timer);

        MPI_Comm comm
            = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
                    A_->getRowMap()->getComm())
                    ->getRawMpiComm());

        // Setup hypredrive.
        HYPRE_CHK_ERR(HYPREDRV_Initialize());
        HYPRE_CHK_ERR(HYPREDRV_Create(comm, &hypredrv_));
        HYPRE_CHK_ERR(HYPREDRV_SetLibraryMode(hypredrv_));
        char* yaml_argv = hypredrive_yaml_.data();
        HYPRE_CHK_ERR(HYPREDRV_InputArgsParse(1, &yaml_argv, hypredrv_));
        HYPRE_CHK_ERR(HYPREDRV_SetGlobalOptions(hypredrv_));

        // Check that RowMap and RangeMap are the same.  While this could
        // handle the case where RowMap and RangeMap are permutations, other
        // Ifpack PCs don't handle this either.
        if (!A_->getRowMap()->isSameAs(*A_->getRangeMap()))
        {
            throw std::runtime_error(
                "Multivectors have different numbers of vectors");
        }

        // Hypre expects the RowMap to be Linear.
        if (A_->getRowMap()->isContiguous())
        {
            globally_contiguous_row_map_ = A_->getRowMap();
            globally_contiguous_col_map_ = A_->getColMap();
        }
        else
        {
            // Must create GloballyContiguous Maps for Hypre
            if (A_->getDomainMap()->isSameAs(*A_->getRowMap()))
            {
                Teuchos::RCP<const Tpetra::CrsMatrix<>> Aconst
                    = Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<>>(A_);
                globally_contiguous_col_map_ = makeContiguousColumnMap(Aconst);
                globally_contiguous_row_map_ = rcp(
                    new Tpetra::Map<>(A_->getRowMap()->getGlobalNumElements(),
                                      A_->getRowMap()->getLocalNumElements(),
                                      0,
                                      A_->getRowMap()->getComm()));
            }
            else
            {
                throw std::runtime_error(
                    "Ifpack_Hypre: Unsupported map configuration: Row/Domain "
                    "maps do not match");
            }
        }

        // Next create vectors that will be used when ApplyInverse() is called
        HYPRE_Int ilower = globally_contiguous_row_map_->getMinGlobalIndex();
        HYPRE_Int iupper = globally_contiguous_row_map_->getMaxGlobalIndex();

        // X in AX = Y
        HYPRE_CHK_ERR(HYPRE_IJVectorCreate(comm, ilower, iupper, &hypre_x_));
        HYPRE_CHK_ERR(HYPRE_IJVectorSetObjectType(hypre_x_, HYPRE_PARCSR));
        HYPRE_CHK_ERR(HYPRE_IJVectorInitialize(hypre_x_));
        HYPRE_CHK_ERR(HYPRE_IJVectorAssemble(hypre_x_));
        HYPRE_CHK_ERR(HYPRE_IJVectorGetObject(hypre_x_, (void**)&par_x_));

        // Y in AX = Y
        HYPRE_CHK_ERR(HYPRE_IJVectorCreate(comm, ilower, iupper, &hypre_y_));
        HYPRE_CHK_ERR(HYPRE_IJVectorSetObjectType(hypre_y_, HYPRE_PARCSR));
        HYPRE_CHK_ERR(HYPRE_IJVectorInitialize(hypre_y_));
        HYPRE_CHK_ERR(HYPRE_IJVectorAssemble(hypre_y_));
        HYPRE_CHK_ERR(HYPRE_IJVectorGetObject(hypre_y_, (void**)&par_y_));

        // Create the Hypre matrix.
        HYPRE_CHK_ERR(HYPRE_IJMatrixCreate(
            comm, ilower, iupper, ilower, iupper, &hypre_A_));
        HYPRE_CHK_ERR(HYPRE_IJMatrixSetObjectType(hypre_A_, HYPRE_PARCSR));
        HYPRE_CHK_ERR(HYPRE_IJMatrixInitialize(hypre_A_));

        // Set hypre matrix with hypredrive
        HYPRE_CHK_ERR(
            HYPREDRV_LinearSystemSetMatrix(hypredrv_, (HYPRE_Matrix)hypre_A_));
        HYPRE_CHK_ERR(HYPREDRV_LinearSystemSetPrecMatrix(hypredrv_));
        HYPRE_CHK_ERR(HYPREDRV_PreconCreate(hypredrv_));

        // set flags
        is_initialized_ = true;
        num_initialize_++;
    }

    // increment timing
    initialize_time_ += (timer->wallTime() - startTime);
}

//==============================================================================
void Hypre::setParameters(const Teuchos::ParameterList& list)
{
    hypredrive_yaml_ = list.get<std::string>("HypreDrive YAML File");
}

//==============================================================================
void Hypre::compute()
{
    const std::string timer_name("LinearSolvers::Hypre::compute");
    Teuchos::RCP<Teuchos::Time> timer
        = Teuchos::TimeMonitor::lookupCounter(timer_name);
    if (timer.is_null())
        timer = Teuchos::TimeMonitor::getNewCounter(timer_name);
    double startTime = timer->wallTime();

    // Scope for timing.
    {
        Teuchos::TimeMonitor time_mon(*timer);

        if (isInitialized() == false)
        {
            initialize();
        }

        // Copy new matrix values to hypre.
        copyTpetraToHypre();

        // Hypredrive setup.
        HYPRE_CHK_ERR(HYPREDRV_PreconSetup(hypredrv_));

        is_computed_ = true;
        num_compute_++;
    }

    compute_time_ += (timer->wallTime() - startTime);
}

//---------------------------------------------------------------------------//
void Hypre::hypreApply(const Tpetra::MultiVector<>& x,
                       Tpetra::MultiVector<>& y) const
{
    if (isComputed() == false)
    {
        throw std::runtime_error("hypre preconditioner not computed");
    }
    hypre_Vector* x_local_ = hypre_ParVectorLocalVector(par_x_);
    hypre_Vector* y_local_ = hypre_ParVectorLocalVector(par_y_);
    size_t num_vectors = x.getNumVectors();

    // X and Y must have same number of vectors
    if (num_vectors != y.getNumVectors())
    {
        throw std::runtime_error(
            "Multivectors have different numbers of vectors");
    }

    // NOTE: Here were assuming that the local ordering of Tpetra's X/Y-vectors
    // and Hypre's X/Y-vectors are the same.
    for (int vec_num = 0; vec_num < (int)num_vectors; vec_num++)
    {
        // Get values for current vector in multivector.
        double* x_values = const_cast<double*>(x.getData(vec_num).getRawPtr());
        double* y_values = const_cast<double*>(y.getData(vec_num).getRawPtr());

        // Temporarily make a pointer to data in Hypre for end
        double* x_temp = x_local_->data;
        double* y_temp = y_local_->data;

        // Replace data in Hypre vectors with Tpetra data
        x_local_->data = x_values;
        y_local_->data = y_values;

        // Reset result vector.
        HYPRE_CHK_ERR(HYPRE_ParVectorSetConstantValues(par_y_, 0.0));

        // Apply the preconditioner.
        HYPRE_CHK_ERR(HYPREDRV_PreconApply(
            hypredrv_, (HYPRE_Vector)hypre_x_, (HYPRE_Vector)hypre_y_));

        // Move data back to hypre.
        x_local_->data = x_temp;
        y_local_->data = y_temp;
    }
}

//==============================================================================
// Apply preconditioner: y = alpha * P * x + beta * y
void Hypre::apply(const Tpetra::MultiVector<>& x,
                  Tpetra::MultiVector<>& y,
                  Teuchos::ETransp mode,
                  double alpha,
                  double beta) const
{
    num_apply_++;

    const std::string timer_name("LinearSolvers::Hypre::apply");
    Teuchos::RCP<Teuchos::Time> timer
        = Teuchos::TimeMonitor::lookupCounter(timer_name);
    if (timer.is_null())
        timer = Teuchos::TimeMonitor::getNewCounter(timer_name);
    double start_time = timer->wallTime();

    // Timing scope.
    {
        Teuchos::TimeMonitor time_mon(*timer);

        if (mode != Teuchos::NO_TRANS)
        {
            throw std::logic_error(
                "VertexCFD preconditioner does not support apply with "
                "transpose");
        }

        // Use Belos class to perform common operations on multivectors
        using MV_Traits = Belos::MultiVecTraits<double, Tpetra::MultiVector<>>;

        assert(MV_Traits::GetNumberVecs(x) == MV_Traits::GetNumberVecs(y));

        // If alpha is zero, don't apply operator, just scale and return
        if (alpha == Teuchos::ScalarTraits<double>::zero())
        {
            MV_Traits::MvScale(y, beta);
            apply_time_ += (timer->wallTime() - start_time);
            return;
        }

        // If beta is zero, do apply and scale
        if (beta == Teuchos::ScalarTraits<double>::zero())
        {
            hypreApply(x, y);
            MV_Traits::MvScale(y, alpha);
        }
        else
        {
            // For nonzero beta, need temporary vector
            Teuchos::RCP<Tpetra::MultiVector<>> z = MV_Traits::Clone(x, 1);
            hypreApply(x, *z);
            MV_Traits::MvAddMv(alpha, *z, beta, y, y);
        }
    }
    apply_time_ += (timer->wallTime() - start_time);
}

//==============================================================================
std::string Hypre::description() const
{
    std::ostringstream out;

    // Output is a valid YAML dictionary in flow style.  If you don't
    // like everything on a single line, you should call describe()
    // instead.
    out << "\"LinearSolvers::Hypre\": {";
    out << "Initialized: " << (isInitialized() ? "true" : "false") << ", "
        << "Computed: " << (isComputed() ? "true" : "false") << ", ";

    if (A_.is_null())
    {
        out << "Matrix: null";
    }
    else
    {
        out << "Global matrix dimensions: [" << A_->getGlobalNumRows() << ", "
            << A_->getGlobalNumCols() << "]"
            << ", Global nnz: " << A_->getGlobalNumEntries();
    }

    out << "}";
    return out.str();
}

//==============================================================================
void Hypre::describe(Teuchos::FancyOStream& os,
                     const Teuchos::EVerbosityLevel) const
{
    using std::endl;
    os << endl;
    os << "==================================================================="
          "============="
       << endl;
    os << "LinearSolvers::Hypre: " << endl << endl;
    os << "Using " << A_->getComm()->getSize() << " processors." << endl;
    os << "Global number of rows            = " << A_->getGlobalNumRows()
       << endl;
    os << "Global number of nonzeros        = " << A_->getGlobalNumEntries()
       << endl;
    //    os << "Condition number estimate = " << Condest() << endl;
    os << endl;
    os << "Phase           # calls   Total Time (s)" << endl;
    os << "-----           -------   --------------" << endl;
    os << "Initialize()    " << std::setw(5) << num_initialize_ << "  "
       << std::setw(15) << initialize_time_ << endl;
    os << "Compute()       " << std::setw(5) << num_compute_ << "  "
       << std::setw(15) << compute_time_ << endl;
    os << "ApplyInverse()  " << std::setw(5) << num_apply_ << "  "
       << std::setw(15) << apply_time_ << endl;
    os << "==================================================================="
          "============="
       << endl;
    os << endl;
}

//==============================================================================
int Hypre::copyTpetraToHypre()
{
    using LO = Tpetra::CrsMatrix<>::local_ordinal_type;
    using GO = Tpetra::CrsMatrix<>::global_ordinal_type;

    Teuchos::RCP<const Tpetra::CrsMatrix<>> matrix
        = Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<>>(A_);
    if (matrix.is_null())
        throw std::runtime_error(
            "Hypre<Tpetra::RowMatrix<double, LocalOrdinal, HYPRE_Int, Node>: "
            "Unsupported matrix configuration: Tpetra::CrsMatrix required");

    std::vector<HYPRE_Int> new_indices(matrix->getLocalMaxNumRowEntries());
    for (LO i = 0; i < (LO)matrix->getLocalNumRows(); i++)
    {
        typename Tpetra::CrsMatrix<>::values_host_view_type values;
        typename Tpetra::CrsMatrix<>::local_inds_host_view_type indices;
        matrix->getLocalRowView(i, indices, values);
        for (LO j = 0; j < (LO)indices.extent(0); j++)
        {
            new_indices[j]
                = globally_contiguous_col_map_->getGlobalElement(indices(j));
        }
        HYPRE_Int GlobalRow[1];
        HYPRE_Int numEntries = (GO)indices.extent(0);
        GlobalRow[0] = globally_contiguous_row_map_->getGlobalElement(i);
        HYPRE_CHK_ERR(HYPRE_IJMatrixSetValues(hypre_A_,
                                              1,
                                              &numEntries,
                                              GlobalRow,
                                              new_indices.data(),
                                              values.data()));
    }
    HYPRE_CHK_ERR(HYPRE_IJMatrixAssemble(hypre_A_));
    return 0;
}

//==============================================================================
Teuchos::RCP<const Tpetra::Map<>> Hypre::makeContiguousColumnMap(
    Teuchos::RCP<const Tpetra::CrsMatrix<>>& matrix) const
{
    using GoVector = Tpetra::Vector<global_ordinal_type,
                                    local_ordinal_type,
                                    global_ordinal_type,
                                    node_type>;

    // Must create globally_contiguous_domain_map (which is a permutation of
    // Matrix A_'s domain_map) and the corresponding permuted column_map.
    //   Tpetra_GID  --------->   LID   ----------> HYPRE_GID
    //           via domain_map.LID()       via
    //           globally_contiguous_domain_map.GID()
    if (matrix.is_null())
        throw std::runtime_error(
            "Hypre<Tpetra::RowMatrix<HYPRE_Real, HYPRE_Int, long long, Node>: "
            "Unsupported matrix configuration: Tpetra::CrsMatrix required");
    RCP<const Tpetra::Map<>> domain_map = matrix->getDomainMap();
    RCP<const Tpetra::Map<>> column_map = matrix->getColMap();
    RCP<const Tpetra::Import<>> importer = matrix->getGraph()->getImporter();

    if (domain_map->isContiguous())
    {
        // If the domain map is linear, then we can just use the column map as
        // is.
        return column_map;
    }
    else
    {
        // The domain map isn't linear, so we need a new domain map
        Teuchos::RCP<Tpetra::Map<>> contiguous_domain_map = Teuchos::rcp(
            new Tpetra::Map<>(domain_map->getGlobalNumElements(),
                              domain_map->getLocalNumElements(),
                              0,
                              domain_map->getComm()));
        if (importer)
        {
            // If there's an importer then we can use it to get a new column
            // map
            GoVector my_gids_hypre(
                domain_map, contiguous_domain_map->getLocalElementList());

            // import the HYPRE GIDs
            GoVector col_gids_hypre(column_map);
            col_gids_hypre.doImport(my_gids_hypre, *importer, Tpetra::INSERT);

            // Make a HYPRE numbering-based column map.
            return Teuchos::rcp(
                new Tpetra::Map<>(column_map->getGlobalNumElements(),
                                  col_gids_hypre.getDataNonConst()(),
                                  0,
                                  column_map->getComm()));
        }
        else
        {
            // The problem has matching domain/column maps, and somehow the
            // domain map isn't linear, so just use the new domain map
            return Teuchos::rcp(
                new Tpetra::Map<>(column_map->getGlobalNumElements(),
                                  contiguous_domain_map->getLocalElementList(),
                                  0,
                                  column_map->getComm()));
        }
    }
}

//---------------------------------------------------------------------------//
int Hypre::getNumInitialize() const
{
    return num_initialize_;
}

//---------------------------------------------------------------------------//
int Hypre::getNumCompute() const
{
    return num_compute_;
}

//---------------------------------------------------------------------------//
int Hypre::getNumApply() const
{
    return num_apply_;
}

//---------------------------------------------------------------------------//
double Hypre::getInitializeTime() const
{
    return initialize_time_;
}

//---------------------------------------------------------------------------//
double Hypre::getComputeTime() const
{
    return compute_time_;
}

//---------------------------------------------------------------------------//
double Hypre::getApplyTime() const
{
    return apply_time_;
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const typename Tpetra::Map<>> Hypre::getDomainMap() const
{
    Teuchos::RCP<const Tpetra::RowMatrix<>> A = getMatrix();
    TEUCHOS_TEST_FOR_EXCEPTION(A.is_null(),
                               std::runtime_error,
                               "LinearSolvers::Hypre::getDomainMap: The "
                               "input matrix A is null.  Please call "
                               "setMatrix() with a nonnull input "
                               "matrix before calling this method.");
    return A->getDomainMap();
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const typename Tpetra::Map<>> Hypre::getRangeMap() const
{
    Teuchos::RCP<const Tpetra::RowMatrix<>> A = getMatrix();
    TEUCHOS_TEST_FOR_EXCEPTION(A.is_null(),
                               std::runtime_error,
                               "LinearSolvers::Hypre::getRangeMap: The "
                               "input matrix A is null.  Please call "
                               "setMatrix() with a nonnull input "
                               "matrix before calling this method.");
    return A->getRangeMap();
}

//---------------------------------------------------------------------------//
void Hypre::setMatrix(const Teuchos::RCP<const Tpetra::RowMatrix<>>& A)
{
    if (A.getRawPtr() != getMatrix().getRawPtr())
    {
        is_initialized_ = false;
        is_computed_ = false;
        A_ = A;
    }
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const typename Tpetra::RowMatrix<>> Hypre::getMatrix() const
{
    return A_;
}

//---------------------------------------------------------------------------//
bool Hypre::hasTransposeApply() const
{
    return false;
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
