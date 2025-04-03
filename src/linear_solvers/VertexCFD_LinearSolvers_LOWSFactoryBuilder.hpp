#ifndef VERTEXCFD_LINEARSOLVERS_LOWSFACTORYBUILDER_HPP
#define VERTEXCFD_LINEARSOLVERS_LOWSFACTORYBUILDER_HPP

#include <Thyra_LinearOpWithSolveFactoryBase.hpp>
#include <Thyra_PreconditionerFactoryBase.hpp>

#include <memory>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Build a "Linear Op With Solve" (i.e., linear solver) factory.
// This class uses the Stratimikos DefaultLinearSolverBuilder, but allows
// custom VertexCFD preconditioners to be used with a Trilinos linear solver.
//---------------------------------------------------------------------------//

class LOWSFactoryBuilder
{
  public:
    // Prevent construction
    LOWSFactoryBuilder() = delete;

    // Build solver from solver name.
    static Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<double>>
    buildLOWS(Teuchos::RCP<Teuchos::ParameterList> params);
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

//---------------------------------------------------------------------------//
#endif // VERTEXCFD_LINEARSOLVERS_LOWSFACTORYBUILDER_HPP
//---------------------------------------------------------------------------//
