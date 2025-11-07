#include "VertexCFD_LinearSolvers_Preconditioner.hpp"
#include "VertexCFD_LinearSolvers_LocalSolverFactory.hpp"

#include <BelosMultiVecTraits.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Teuchos_TimeMonitor.hpp>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Constructor
//---------------------------------------------------------------------------//
Preconditioner::Preconditioner()
    : _initialized(false)
    , _computed(false)
    , _num_initialize(0)
    , _num_compute(0)
    , _num_apply(0)
    , _init_time(0.0)
    , _compute_time(0.0)
    , _apply_time(0.0)
{
}

//---------------------------------------------------------------------------//
// Set parameters governing solve and construct local solver
//---------------------------------------------------------------------------//
void Preconditioner::setParameters(const Teuchos::ParameterList& params)
{
    _local_solver = LocalSolverFactory::buildSolver(params);
}

//---------------------------------------------------------------------------//
// Change matrix
//---------------------------------------------------------------------------//
void Preconditioner::setMatrix(const Teuchos::RCP<const Tpetra::RowMatrix<>>& A)
{
    auto timer = Teuchos::TimeMonitor::getNewTimer(_set_label);
    const Teuchos::TimeMonitor tm(*timer);

    // Depending on order of construction of AdditiveSchwarz, the matrix
    //  may be null on the first call to setMatrix. In this case, setMatrix
    //  will be called again later.
    _A = A;
    if (!_A)
        return;

    _local_solver->setMatrix(_A);
    _computed = false;
}

//---------------------------------------------------------------------------//
// Initialize preconditioner (symbolic factorization)
//---------------------------------------------------------------------------//
void Preconditioner::initialize()
{
    auto timer = Teuchos::TimeMonitor::getNewTimer(_init_label);
    const Teuchos::TimeMonitor tm(*timer);
    const double start_time = Teuchos::Time::wallTime();

    _local_solver->initialize();
    _initialized = true;
    _num_initialize++;
    _init_time += (Teuchos::Time::wallTime() - start_time);
}

//---------------------------------------------------------------------------//
// Compute preconditioner (numeric factorization)
//---------------------------------------------------------------------------//
void Preconditioner::compute()
{
    auto timer = Teuchos::TimeMonitor::getNewTimer(_compute_label);
    const Teuchos::TimeMonitor tm(*timer);
    const double start_time = Teuchos::Time::wallTime();

    _local_solver->compute();
    _computed = true;
    _num_compute++;
    _compute_time += (Teuchos::Time::wallTime() - start_time);
}

//---------------------------------------------------------------------------//
// Apply preconditioner: y = alpha * P * x + beta * y
//---------------------------------------------------------------------------//
void Preconditioner::apply(const Tpetra::MultiVector<>& x,
                           Tpetra::MultiVector<>& y,
                           Teuchos::ETransp mode,
                           double alpha,
                           double beta) const
{
    auto timer = Teuchos::TimeMonitor::getNewTimer(_apply_label);
    const Teuchos::TimeMonitor tm(*timer);
    const double start_time = Teuchos::Time::wallTime();

    _num_apply++;

    if (mode != Teuchos::NO_TRANS)
    {
        throw std::logic_error(
            "VertexCFD preconditioner does not support apply with transpose");
    }

    // Use Belos class to perform common operations on multivectors
    using MV_Traits = Belos::MultiVecTraits<double, Tpetra::MultiVector<>>;

    assert(MV_Traits::GetNumberVecs(x) == MV_Traits::GetNumberVecs(y));

    // If alpha is zero, don't apply operator, just scale and return
    if (alpha == Teuchos::ScalarTraits<double>::zero())
    {
        MV_Traits::MvScale(y, beta);
        _apply_time += (Teuchos::Time::wallTime() - start_time);
        return;
    }

    // If beta is zero, do apply and scale
    if (beta == Teuchos::ScalarTraits<double>::zero())
    {
        _local_solver->solve(x, y);
        MV_Traits::MvScale(y, alpha);
    }
    else
    {
        // For nonzero beta, need temporary vector
        const Teuchos::RCP<Tpetra::MultiVector<>> z = MV_Traits::Clone(x, 1);
        _local_solver->solve(x, *z);

        MV_Traits::MvAddMv(alpha, *z, beta, y, y);
    }
    _apply_time += (Teuchos::Time::wallTime() - start_time);
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
