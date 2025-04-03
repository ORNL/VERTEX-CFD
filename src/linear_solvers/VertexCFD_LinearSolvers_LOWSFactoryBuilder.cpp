#include "VertexCFD_LinearSolvers_LOWSFactoryBuilder.hpp"
#include "VertexCFD_LinearSolvers_PreconditionerFactory.hpp"

#ifdef VERTEXCFD_HAVE_HYPRE
#include "VertexCFD_LinearSolvers_HyprePreconditionerFactory.hpp"
#endif

#include <PanzerCore_config.hpp>
#include <Panzer_NodeType.hpp>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include <Teuchos_AbstractFactoryStd.hpp>
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#include <Thyra_PreconditionerFactoryBase.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Trilinos_version.h>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Build a linear op with solve
//---------------------------------------------------------------------------//
Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<double>>
LOWSFactoryBuilder::buildLOWS(Teuchos::RCP<Teuchos::ParameterList> params)
{
    // The default Stratimikos solver builder recognizes Ifpack and ML
    // preconditioners, but Ifpack2 and MueLu must be explicitly registered
    Stratimikos::DefaultLinearSolverBuilder builder;

    {
        using Base = Thyra::PreconditionerFactoryBase<double>;
        using Impl = Thyra::Ifpack2PreconditionerFactory<
            Tpetra::CrsMatrix<double, int, panzer::GlobalOrdinal, panzer::TpetraNodeType>>;

        builder.setPreconditioningStrategyFactory(
            Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
    }

    {
#if TRILINOS_MAJOR_MINOR_VERSION >= 130500
        Stratimikos::enableMueLu<double,
                                 int,
                                 panzer::GlobalOrdinal,
                                 panzer::TpetraNodeType>(builder, "MueLu");
#else
        Stratimikos::enableMueLu<int, panzer::GlobalOrdinal, panzer::TpetraNodeType>(
            builder, "MueLu");
#endif
    }

    {
        // Register VertexCFD preconditioner factory with Stratimikos
        using Base = Thyra::PreconditionerFactoryBase<double>;
        using Impl = VertexCFD::LinearSolvers::PreconditionerFactory;
        builder.setPreconditioningStrategyFactory(
            Teuchos::abstractFactoryStd<Base, Impl>(), "VertexCFD");
    }

#ifdef VERTEXCFD_HAVE_HYPRE
    {
        // Register HYPRE preconditioner factory with Stratimikos
        using Base = Thyra::PreconditionerFactoryBase<double>;
        using Impl = VertexCFD::LinearSolvers::HyprePreconditionerFactory;
        builder.setPreconditioningStrategyFactory(
            Teuchos::abstractFactoryStd<Base, Impl>(), "Hypre");
    }
#endif

    builder.setParameterList(params);
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<double>> lowsFactory
        = createLinearSolveStrategy(builder);

    return lowsFactory;
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
