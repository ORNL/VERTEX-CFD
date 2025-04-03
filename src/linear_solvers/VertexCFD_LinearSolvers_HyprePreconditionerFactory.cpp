#include "VertexCFD_LinearSolvers_HyprePreconditionerFactory.hpp"
#include "VertexCFD_LinearSolvers_Hypre.hpp"

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
bool HyprePreconditionerFactory::isCompatible(
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
HyprePreconditionerFactory::createPrec() const
{
    return Teuchos::rcp(new Thyra::DefaultPreconditioner<double>());
}

//---------------------------------------------------------------------------//
// Initialize preconditioner
//---------------------------------------------------------------------------//
void HyprePreconditionerFactory::initializePrec(
    const Teuchos::RCP<const Thyra::LinearOpSourceBase<double>>& fwd_op_src,
    Thyra::PreconditionerBase<double>* prec_op,
    const Thyra::ESupportSolveUse /* not used */) const

{
    //
    // Extract raw Tpetra matrix
    //
    auto fwd_op = fwd_op_src->getOp();
    auto thyra_tpetra_op = Teuchos::rcp_dynamic_cast<
        const Thyra::TpetraLinearOp<double,
                                    int,
                                    panzer::GlobalOrdinal,
                                    panzer::TpetraNodeType>>(fwd_op);
    auto tpetra_op = thyra_tpetra_op->getConstTpetraOperator();
    auto tpetra_row_mat
        = Teuchos::rcp_dynamic_cast<const Tpetra::RowMatrix<>>(tpetra_op);

    // Build Hypre preconditioner and set data.
    _hypre = Teuchos::rcp(new Hypre());
    _hypre->setParameters(*_params);
    Teuchos::rcp_dynamic_cast<Hypre>(_hypre)->setMatrix(tpetra_row_mat);

    // Wrap hypre into a Thyra::Preconditioner
    auto thyra_hypre = Thyra::createLinearOp<double,
                                             int,
                                             panzer::GlobalOrdinal,
                                             panzer::TpetraNodeType>(_hypre);

    // Cast input arg to Thyra::DefaultPreconditioner and set operator
    auto* default_prec
        = dynamic_cast<Thyra::DefaultPreconditioner<double>*>(prec_op);
    default_prec->initializeUnspecified(thyra_hypre);

    // Initialize and compute preconditioner.
    _hypre->initialize();
    _hypre->compute();
    return;
}

//---------------------------------------------------------------------------//
// Uninitialize preconditioner
//---------------------------------------------------------------------------//
void HyprePreconditionerFactory::uninitializePrec(
    Thyra::PreconditionerBase<double>*,
    Teuchos::RCP<const Thyra::LinearOpSourceBase<double>>*,
    Thyra::ESupportSolveUse*) const
{
}

//---------------------------------------------------------------------------//
// Set parameters
//---------------------------------------------------------------------------//
void HyprePreconditionerFactory::setParameterList(
    const Teuchos::RCP<Teuchos::ParameterList>& params)
{
    _params = params;
}

//---------------------------------------------------------------------------//
// Return ParameterList
//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
HyprePreconditionerFactory::getNonconstParameterList()
{
    return _params;
}

//---------------------------------------------------------------------------//
// Clear existing parameters
//---------------------------------------------------------------------------//
Teuchos::RCP<Teuchos::ParameterList>
HyprePreconditionerFactory::unsetParameterList()
{
    auto old_params = _params;
    _params = Teuchos::null;
    return old_params;
}

//---------------------------------------------------------------------------//
// Get valid parameters
//---------------------------------------------------------------------------//
Teuchos::RCP<const Teuchos::ParameterList>
HyprePreconditionerFactory::getValidParameters() const
{
    auto params = Teuchos::rcp(new Teuchos::ParameterList());
    params->set("HypreDrive YAML File", "");
    return params;
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
