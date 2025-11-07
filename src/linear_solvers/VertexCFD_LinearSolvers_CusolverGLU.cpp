#include "VertexCFD_LinearSolvers_CusolverGLU.hpp"

#include <cassert>
#include <sstream>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Constructor
//---------------------------------------------------------------------------//
CusolverGLU::CusolverGLU(const Teuchos::ParameterList& params)
    : _matrix_set(false)
    , _initialized(false)
    , _computed(false)
{
    // Get factorization parameters
    _pivot_threshold = 0.01;
    if (params.isType<double>("Pivot Threshold"))
        _pivot_threshold = params.get<double>("Pivot Threshold");

    // Reorder parameter, 0 is no reordering, 1 uses METIS
    _reorder = 1;
    if (params.isType<int>("Reorder"))
        _reorder = params.get<int>("Reorder");
}

//---------------------------------------------------------------------------//
// Change matrix
//---------------------------------------------------------------------------//
void CusolverGLU::setMatrix(Teuchos::RCP<const Tpetra::RowMatrix<>> A)
{
    _A = A;

    // Host-side allocation for CRS data
    _A_rowptr_host.clear();
    _A_colind_host.clear();
    _A_values_host.clear();

    // Extract matrix data for factorizations
    int local_rows = _A->getLocalNumRows();
    std::size_t num_entries;
    std::size_t max_entries = _A->getLocalMaxNumRowEntries();
    Tpetra::RowMatrix<>::nonconst_local_inds_host_view_type row_inds(
        "row_indices", max_entries);
    Tpetra::RowMatrix<>::nonconst_values_host_view_type row_vals("row_values",
                                                                 max_entries);
    _A_rowptr_host.push_back(0);
    for (int row = 0; row < local_rows; ++row)
    {
        num_entries = _A->getNumEntriesInLocalRow(row);
        _A->getLocalRowCopy(row, row_inds, row_vals, num_entries);
        _A_colind_host.insert(_A_colind_host.end(),
                              row_inds.data(),
                              row_inds.data() + num_entries);
        _A_values_host.insert(_A_values_host.end(),
                              row_vals.data(),
                              row_vals.data() + num_entries);
        _A_rowptr_host.push_back(_A_rowptr_host.back() + num_entries);
    }

    // Copy data to device
    _A_rowptr = _A_rowptr_host;
    _A_colind = _A_colind_host;
    _A_values = _A_values_host;

    _matrix_set = true;
    _computed = false;
}

//---------------------------------------------------------------------------//
// Initialize preconditioner
//---------------------------------------------------------------------------//
void CusolverGLU::initialize()
{
    assert(_matrix_set);
    assert(!_computed);

    if (_initialized)
    {
        return;
    }

    // In future Trilinos versions (13.4+), it might be possible to access the
    // internal matrix data on both host and device without making extra
    // copies. For now, the only way to get the data is to extract the values
    // row-by-row on the host and then explicitly copy to the device.

    int num_rows = _A->getLocalNumRows();
    int A_nnz = _A->getLocalNumEntries();

    cusolverStatus_t stat;
    cusolverSpCreate(&_handle);
    cusparseCreateMatDescr(&_A_descr);
    cusparseSetMatType(_A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(_A_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusolverSpCreateCsrluInfoHost(&_lu_info);

    // Allocate pivot data
    std::vector<int> P(num_rows);
    std::vector<int> Q(num_rows);

    stat = cusolverSpXcsrluConfigHost(_lu_info, _reorder);
    check_status(stat, "cuSOLVER CSR config");

    // Host-side analysis
    stat = cusolverSpXcsrluAnalysisHost(_handle,
                                        num_rows,
                                        A_nnz,
                                        _A_descr,
                                        _A_rowptr_host.data(),
                                        _A_colind_host.data(),
                                        _lu_info);
    check_status(stat, "cuSOLVER host analysis");

    size_t internalDataInBytes;
    size_t workspaceInBytes;
    stat = cusolverSpDcsrluBufferInfoHost(_handle,
                                          num_rows,
                                          A_nnz,
                                          _A_descr,
                                          _A_values_host.data(),
                                          _A_rowptr_host.data(),
                                          _A_colind_host.data(),
                                          _lu_info,
                                          &internalDataInBytes,
                                          &workspaceInBytes);
    check_status(stat, "cuSOLVER buffer info");

    std::vector<char> h_work(workspaceInBytes);
    stat = cusolverSpDcsrluFactorHost(_handle,
                                      num_rows,
                                      A_nnz,
                                      _A_descr,
                                      _A_values_host.data(),
                                      _A_rowptr_host.data(),
                                      _A_colind_host.data(),
                                      _lu_info,
                                      _pivot_threshold,
                                      h_work.data());
    check_status(stat, "cuSOLVER host factorization");

    // Prepare to extract M (compressed LU factors)
    cusolverSpCreateGluInfo(&_M_info);
    cusparseCreateMatDescr(&_M_descr);
    cusparseSetMatType(_M_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(_M_descr, CUSPARSE_INDEX_BASE_ZERO);

    int M_nnz;
    stat = cusolverSpXcsrluNnzMHost(_handle, &M_nnz, _lu_info);
    assert(CUSOLVER_STATUS_SUCCESS == stat);
    check_status(stat, "cuSOLVER CSR nnz");

    std::vector<int> M_rowptr(num_rows + 1);
    std::vector<int> M_colind(M_nnz);

    stat = cusolverSpDcsrluExtractMHost(_handle,
                                        P.data(),
                                        Q.data(),
                                        _M_descr,
                                        NULL, /* csrValM */
                                        M_rowptr.data(),
                                        M_colind.data(),
                                        _lu_info,
                                        h_work.data());
    check_status(stat, "cuSOLVER CSR extract");

    // Set up GLU using host data
    stat = cusolverSpDgluSetup(_handle,
                               num_rows,
                               A_nnz,
                               _A_descr,
                               _A_rowptr_host.data(),
                               _A_colind_host.data(),
                               P.data(),
                               Q.data(),
                               M_nnz,
                               _M_descr,
                               M_rowptr.data(),
                               M_colind.data(),
                               _M_info);
    check_status(stat, "GLU setup");

    size_t buffer_size;
    stat = cusolverSpDgluBufferSize(_handle, _M_info, &buffer_size);
    check_status(stat, "GLU buffer allocation");

    _work.resize(buffer_size);
    stat = cusolverSpDgluAnalysis(_handle, _M_info, _work.data().get());
    check_status(stat, "GLU analysis");

    _initialized = true;
}

//---------------------------------------------------------------------------//
// Compute preconditioner
//---------------------------------------------------------------------------//
void CusolverGLU::compute()
{
    assert(_matrix_set);
    assert(_initialized);
    assert(!_computed);

    int num_rows = _A->getLocalNumRows();
    int A_nnz = _A->getLocalNumEntries();

    cusolverStatus_t stat;
    stat = cusolverSpDgluReset(_handle,
                               num_rows,
                               A_nnz,
                               _A_descr,
                               _A_values.data().get(),
                               _A_rowptr.data().get(),
                               _A_colind.data().get(),
                               _M_info);
    check_status(stat, "GLU reset");

    stat = cusolverSpDgluFactor(_handle, _M_info, _work.data().get());
    check_status(stat, "GLU refactor");

    _computed = true;
}

//---------------------------------------------------------------------------//
// Apply preconditioner
//---------------------------------------------------------------------------//
void CusolverGLU::solve(const Tpetra::MultiVector<>& b,
                        Tpetra::MultiVector<>& x)
{
    cudaDeviceSynchronize();

    assert(_initialized);
    assert(_computed);

    int num_rows = _A->getLocalNumRows();
    int A_nnz = _A->getLocalNumEntries();

    // Get Kokkos Views
    auto b_view = b.getLocalViewDevice(Tpetra::Access::ReadOnly);
    auto x_view = x.getLocalViewDevice(Tpetra::Access::OverwriteAll);

    cusolverStatus_t stat;
    int ite_refine_succ = 0;
    double r_nrminf;
    stat = cusolverSpDgluSolve(_handle,
                               num_rows,
                               A_nnz,
                               _A_descr,
                               _A_values.data().get(),
                               _A_rowptr.data().get(),
                               _A_colind.data().get(),
                               b_view.data(),
                               x_view.data(),
                               &ite_refine_succ,
                               &r_nrminf,
                               _M_info,
                               _work.data().get());
    check_status(stat, "GLU solve");
    if (ite_refine_succ != 1)
    {
        std::stringstream ss;
        ss << "GLU iterative refinement failed with status " << ite_refine_succ;
        throw std::runtime_error(ss.str());
    }
}

//---------------------------------------------------------------------------//
// Apply preconditioner
//---------------------------------------------------------------------------//
void CusolverGLU::check_status(cusolverStatus_t stat,
                               const std::string& msg) const
{
    cudaDeviceSynchronize();
    if (stat == CUSOLVER_STATUS_ALLOC_FAILED)
    {
        std::stringstream ss;
        ss << msg << " failed to allocate device memory" << stat;
        throw std::runtime_error(ss.str());
    }
    else if (stat != CUSOLVER_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << msg << " failed with status " << stat;
        throw std::runtime_error(ss.str());
    }
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
