#include "VertexCFD_NOXObserver_IterationOutput.hpp"

#include <NOX_SolverStats.hpp>

#include <iostream>

namespace VertexCFD
{
namespace NOXObserver
{
//---------------------------------------------------------------------------//
IterationOutput::IterationOutput()
    : _ostream(Teuchos::rcp(&std::cout, false))
{
    _ostream.setShowProcRank(false);
    _ostream.setOutputToRootOnly(0);
}

//---------------------------------------------------------------------------//
void IterationOutput::runPreIterate(const NOX::Solver::Generic&) {}

//---------------------------------------------------------------------------//
void IterationOutput::runPostIterate(const NOX::Solver::Generic& solver)
{
    // Output nonlinear solver results after nonlinear iteration.
    const auto stats = solver.getSolverStatistics();
    const auto& group = solver.getSolutionGroup();
    const SolveDataExtractor& data_extractor
        = dynamic_cast<const NOX::Thyra::Group&>(group);
    _ostream << "   " << std::setw(9) << std::right << std::fixed
             << stats->numNonlinearIterations;
    _ostream << "   " << std::setw(8) << std::left << std::setprecision(2)
             << std::scientific << group.getNormF();
    _ostream << "   " << std::setw(8) << std::right << std::fixed
             << stats->linearSolve.lastLinearSolve_NumIterations;
    _ostream << "   " << std::setw(8) << std::left << std::setprecision(2)
             << std::scientific
             << stats->linearSolve.lastLinearSolve_AchievedTolerance;
    _ostream << "   \n";
}

//---------------------------------------------------------------------------//
void IterationOutput::runPreSolve(const NOX::Solver::Generic& solver)
{
    _ostream << " | ";
    _ostream << std::setw(9) << std::right << "Nonlinear";
    _ostream << " | " << std::setw(8) << std::left << "F 2-Norm";
    _ostream << " | ";
    _ostream << std::setw(8) << std::right << "# Linear";
    _ostream << " | " << std::setw(8) << std::left << "R 2-Norm";
    _ostream << " | \n";

    const auto& group = solver.getSolutionGroup();
    if (!group.isF())
    {
        const_cast<NOX::Abstract::Group&>(group).computeF();
    }
    _ostream << "   " << std::setw(9) << std::right << std::fixed << 0;
    _ostream << "   " << std::setw(8) << std::left << std::setprecision(2)
             << std::scientific << group.getNormF() << "\n";
}

//---------------------------------------------------------------------------//
void IterationOutput::runPostSolve(const NOX::Solver::Generic&) {}

//---------------------------------------------------------------------------//
SolveDataExtractor::SolveDataExtractor(const NOX::Thyra::Group& group)
    : NOX::Thyra::Group(group, NOX::ShapeCopy)
{
}

//---------------------------------------------------------------------------//
int SolveDataExtractor::lastLinearSolveNumIters() const
{
    return this->last_linear_solve_num_iters_;
}

//---------------------------------------------------------------------------//
double SolveDataExtractor::lastLinearSolveAchievedTol() const
{
    return this->last_linear_solve_achieved_tol_;
}

//---------------------------------------------------------------------------//

} // end namespace NOXObserver
} // end namespace VertexCFD
