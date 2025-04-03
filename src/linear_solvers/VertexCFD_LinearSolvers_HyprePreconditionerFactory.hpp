#ifndef VERTEXCFD_LINEARSOLVERS_HYPREPRECONDITIONERFACTORY_HPP
#define VERTEXCFD_LINEARSOLVERS_HYPREPRECONDITIONERFACTORY_HPP

#include <Ifpack2_Preconditioner.hpp>

#include <Thyra_LinearOpWithSolveFactoryBase.hpp>
#include <Thyra_PreconditionerFactoryBase.hpp>

#include <memory>

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Build a Hypre Preconditioner.
// This class allows for the construction of "custom" preconditioners that
// aren't supported through Trilinos.
//---------------------------------------------------------------------------//
class HyprePreconditionerFactory
    : public Thyra::PreconditionerFactoryBase<double>
{
  public:
    // Determine if preconditioner is compatible with specified operator
    bool isCompatible(
        const Thyra::LinearOpSourceBase<double>& fwdOpSrc) const override;

    // Construct (but do not initialize) preconditioner
    Teuchos::RCP<Thyra::PreconditionerBase<double>> createPrec() const override;

    // Initialize preconditioner
    void initializePrec(
        const Teuchos::RCP<const Thyra::LinearOpSourceBase<double>>& fwdOpSrc,
        Thyra::PreconditionerBase<double>* precOp,
        const Thyra::ESupportSolveUse supportSolveUse
        = Thyra::SUPPORT_SOLVE_UNSPECIFIED) const override;

    void uninitializePrec(
        Thyra::PreconditionerBase<double>* prec,
        Teuchos::RCP<const Thyra::LinearOpSourceBase<double>>* fwdOpSrc = NULL,
        Thyra::ESupportSolveUse* supportSolveUse = NULL) const override;

    //
    // Teuchos::ParameterListAcceptor API
    //
    void setParameterList(
        const Teuchos::RCP<Teuchos::ParameterList>& params) override;
    Teuchos::RCP<Teuchos::ParameterList> getNonconstParameterList() override;
    Teuchos::RCP<Teuchos::ParameterList> unsetParameterList() override;

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidParameters() const override;

  private:
    Teuchos::RCP<Teuchos::ParameterList> _params;

    mutable Teuchos::RCP<Ifpack2::Preconditioner<>> _hypre;
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif // VERTEXCFD_LINEARSOLVERS_HYPREPRECONDITIONERFACTORY_HPP
