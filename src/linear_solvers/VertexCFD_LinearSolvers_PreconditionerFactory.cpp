#include "VertexCFD_LinearSolvers_PreconditionerFactory.hpp"
#include "VertexCFD_LinearSolvers_Preconditioner.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_NodeType.hpp>
#include <Thyra_DefaultPreconditioner.hpp>
#include <Thyra_TpetraLinearOp.hpp>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Determine if preconditioner is compatible with specified operator
//---------------------------------------------------------------------------//
bool PreconditionerFactory::isCompatible(
    const Thyra::LinearOpSourceBase<double>& fwd_op_src) const
{
    auto fwd_op = fwd_op_src.getOp();

    auto thyra_tpetra_op = Teuchos::rcp_dynamic_cast<
        const Thyra::TpetraLinearOp<double,
                                    int,
                                    panzer::GlobalOrdinal,
                                    panzer::TpetraNodeType>>(fwd_op);
    if (Teuchos::is_null(thyra_tpetra_op))
        return false;

    auto tpetra_op = thyra_tpetra_op->getConstTpetraOperator();
    auto tpetra_row_mat
        = Teuchos::rcp_dynamic_cast<const Tpetra::RowMatrix<>>(tpetra_op);

    return Teuchos::nonnull(tpetra_row_mat);
}

//---------------------------------------------------------------------------//
// Construct (but do not initialize) preconditioner
//---------------------------------------------------------------------------//
Teuchos::RCP<Thyra::PreconditionerBase<double>>
PreconditionerFactory::createPrec() const
{
    return Teuchos::rcp(new Thyra::DefaultPreconditioner<double>());
}

//---------------------------------------------------------------------------//
// Initialize preconditioner
//---------------------------------------------------------------------------//
void PreconditionerFactory::initializePrec(
    const Teuchos::RCP<const Thyra::LinearOpSourceBase<double>>& fwd_op_src,
    Thyra::PreconditionerBase<double>* prec_op,
    const Thyra::ESupportSolveUse /* not used */) const

{
    //
    // Extract raw Tpetra matrix
    //

    using RowMatrix = Tpetra::RowMatrix<>;

    auto fwd_op = fwd_op_src->getOp();
    auto thyra_tpetra_op = Teuchos::rcp_dynamic_cast<
        const Thyra::TpetraLinearOp<double,
                                    int,
                                    panzer::GlobalOrdinal,
                                    panzer::TpetraNodeType>>(fwd_op);
    auto tpetra_op = thyra_tpetra_op->getConstTpetraOperator();
    auto tpetra_row_mat = Teuchos::rcp_dynamic_cast<const RowMatrix>(tpetra_op);

    // Build Additive Schwarz preconditioner
    if (!_schwarz)
    {
        // Build "outer" preconditiner
        _schwarz = Teuchos::rcp(
            new Ifpack2::AdditiveSchwarz<RowMatrix>(tpetra_row_mat));
        _schwarz->setParameters(*_params);

        // Build "inner" preconditioner and compute
        auto inner_prec_params = Teuchos::sublist(
            _params, "schwarz: inner preconditioner parameters");
        auto inner_prec = Teuchos::rcp(new Preconditioner());
        inner_prec->setParameters(*inner_prec_params);
        _schwarz->setInnerPreconditioner(inner_prec);
        _schwarz->initialize();
        _schwarz->compute();

        // Wrap additive Schwarz into a Thyra::Preconditioner
        auto thyra_schwarz
            = Thyra::createLinearOp<double,
                                    int,
                                    panzer::GlobalOrdinal,
                                    panzer::TpetraNodeType>(_schwarz);

        // Cast input arg to Thyra::DefaultPreconditioner and set operator
        auto* default_prec
            = dynamic_cast<Thyra::DefaultPreconditioner<double>*>(prec_op);
        default_prec->initializeUnspecified(thyra_schwarz);
    }
    else
    {
        _schwarz->setMatrix(tpetra_row_mat);
        _schwarz->initialize();
        _schwarz->compute();
        return;
    }
}

//---------------------------------------------------------------------------//
// Uninitialize preconditioner
//---------------------------------------------------------------------------//
void PreconditionerFactory::uninitializePrec(
    Thyra::PreconditionerBase<double>*,
    Teuchos::RCP<const Thyra::LinearOpSourceBase<double>>*,
    Thyra::ESupportSolveUse*) const
{
}

//---------------------------------------------------------------------------//
// Set parameters
//---------------------------------------------------------------------------//
void PreconditionerFactory::setParameterList(
    const Teuchos::RCP<Teuchos::ParameterList>& params)
{
    _params = params;
}

//---------------------------------------------------------------------------//
// Return ParameterList
//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
PreconditionerFactory::getNonconstParameterList()
{
    return _params;
}

//---------------------------------------------------------------------------//
// Clear existing parameters
//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList> PreconditionerFactory::unsetParameterList()
{
    auto old_params = _params;
    _params = Teuchos::null;
    return old_params;
}

//---------------------------------------------------------------------------//
// Get valid parameters
//---------------------------------------------------------------------------//
Teuchos::RCP<const Teuchos::ParameterList>
PreconditionerFactory::getValidParameters() const
{
    // Get parameters for AdditiveSchwarz first
    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    if (_schwarz)
    {
        *params = *(_schwarz->getValidParameters());
    }
    else
    {
        const Teuchos::RCP<Tpetra::RowMatrix<>> row_mat;
        auto schwarz = Teuchos::rcp(
            new Ifpack2::AdditiveSchwarz<Tpetra::RowMatrix<>>(row_mat));
        *params = *(schwarz->getValidParameters());
    }

    // Add VertexCFD preconditioner parameters for inner solver
    auto inner_params
        = Teuchos::sublist(params, "schwarz: inner preconditioner parameters");
    inner_params->set("Local Solver", "Cusolver GLU");
    inner_params->set("Reorder", 1);
    inner_params->set("Pivot Threshold", 1.0e-2);

    return params;
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
