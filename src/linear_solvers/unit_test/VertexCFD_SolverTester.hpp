#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Teuchos_DefaultMpiComm.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
class SolverTester : public ::testing::Test
{
  protected:
    void SetUp()
    {
        using GO = Tpetra::Details::DefaultTypes::global_ordinal_type;
        using LO = Tpetra::Details::DefaultTypes::local_ordinal_type;

        auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
        LO nx = 10;
        LO ny = 10;
        GO num_global_entries = nx * ny;
        LO index_base = 0;
        _map = Teuchos::rcp(
            new Tpetra::Map<>(num_global_entries, index_base, comm));

        LO entries_per_row = 5;
        _matrix = Teuchos::rcp(new Tpetra::CrsMatrix<>(_map, entries_per_row));

        // Build 2D Laplacian with Dirichlet BCs
        Teuchos::ArrayRCP<GO> row_inds(entries_per_row);
        Teuchos::ArrayRCP<double> row_vals(entries_per_row);
        LO num_local_rows = _map->getLocalNumElements();
        for (LO local_row = 0; local_row < num_local_rows; ++local_row)
        {
            GO global_row = _map->getGlobalElement(local_row);
            GO ix = global_row % nx;
            GO iy = global_row / nx;

            // Diagonal entry
            row_inds[0] = global_row;
            row_vals[0] = 4.0;
            int num_entries = 1;

            if (ix > 0)
            {
                row_inds[num_entries] = ix - 1 + iy * nx;
                row_vals[num_entries] = -1.0;
                num_entries++;
            }
            if (ix < (nx - 1))
            {
                row_inds[num_entries] = ix + 1 + iy * nx;
                row_vals[num_entries] = -1.0;
                num_entries++;
            }
            if (iy > 0)
            {
                row_inds[num_entries] = ix + (iy - 1) * nx;
                row_vals[num_entries] = -1.0;
                num_entries++;
            }
            if (iy < (ny - 1))
            {
                row_inds[num_entries] = ix + (iy + 1) * nx;
                row_vals[num_entries] = -1.0;
                num_entries++;
            }

            _matrix->insertGlobalValues(global_row,
                                        row_inds.view(0, num_entries),
                                        row_vals.view(0, num_entries));
        }
        _matrix->fillComplete();

        // Build RHS and solution vector
        _x = Teuchos::rcp(new Tpetra::MultiVector<>(_map, 1));
        _x->putScalar(1.0);
        _y = Teuchos::rcp(new Tpetra::MultiVector<>(_map, 1));
        _y->putScalar(0.0);

        // Store reference solution -- this is result of solving a linear
        // system with above matrix with a right hand side of all ones.
        // Solution computed using numpy
        _ref_soln
            = {1.342423770482685,  2.18484754096537,   2.723654789583171,
               3.0479959507414374, 3.201077948227338,  3.201077948227339,
               3.047995950741439,  2.723654789583172,  2.1848475409653703,
               1.3424237704826851, 2.1848475409653694, 3.6733116037956237,
               4.6617756666258785, 5.267251065155242,  5.5552378939405775,
               5.555237893940578,  5.267251065155244,  4.661775666625878,
               3.6733116037956237, 2.1848475409653707, 2.723654789583172,
               4.661775666625879,  5.982885207969479,  6.803994749313078,
               7.19738466843915,   7.197384668439149,  6.803994749313074,
               5.982885207969476,  4.6617756666258785, 2.7236547895831706,
               3.0479959507414387, 5.267251065155243,  6.803994749313077,
               7.768458055688435,  8.232921362063793,  8.232921362063792,
               7.76845805568843,   6.803994749313073,  5.26725106515524,
               3.047995950741437,  3.2010779482273386, 5.555237893940577,
               7.19738466843915,   8.232921362063792,  8.732921362063793,
               8.732921362063792,  8.232921362063788,  7.197384668439146,
               5.555237893940573,  3.2010779482273364, 3.201077948227339,
               5.555237893940577,  7.197384668439149,  8.23292136206379,
               8.73292136206379,   8.732921362063788,  8.232921362063786,
               7.197384668439144,  5.555237893940571,  3.201077948227336,
               3.0479959507414396, 5.267251065155243,  6.803994749313074,
               7.76845805568843,   8.232921362063788,  8.232921362063786,
               7.768458055688429,  6.80399474931307,   5.267251065155238,
               3.0479959507414365, 2.723654789583173,  4.661775666625879,
               5.982885207969477,  6.803994749313072,  7.1973846684391445,
               7.197384668439141,  6.8039947493130715, 5.982885207969472,
               4.661775666625878,  2.72365478958317,   2.1848475409653707,
               3.6733116037956255, 4.661775666625879,  5.267251065155241,
               5.555237893940573,  5.555237893940573,  5.267251065155241,
               4.661775666625878,  3.673311603795623,  2.18484754096537,
               1.3424237704826854, 2.1848475409653703, 2.7236547895831715,
               3.047995950741438,  3.201077948227337,  3.2010779482273364,
               3.0479959507414374, 2.7236547895831706, 2.1848475409653694,
               1.342423770482685};
    }

    void TearDown() {}

    void solve() {}

  protected:
    Teuchos::RCP<Tpetra::Map<>> _map;
    Teuchos::RCP<Tpetra::CrsMatrix<>> _matrix;
    Teuchos::RCP<Tpetra::MultiVector<>> _x;
    Teuchos::RCP<Tpetra::MultiVector<>> _y;
    std::vector<double> _ref_soln;
};

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
