// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef VERTEXFCFD_LINEARSOLVERS_HYPRE_HPP
#define VERTEXFCFD_LINEARSOLVERS_HYPRE_HPP

#include <Ifpack2_Details_CanChangeMatrix.hpp>
#include <Ifpack2_Preconditioner.hpp>

#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>

#include <HYPREDRV.h>

#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#include <_hypre_IJ_mv.h>
#include <_hypre_parcsr_mv.h>

namespace VertexCFD
{
namespace LinearSolvers
{

//---------------------------------------------------------------------------//
class Hypre : public Ifpack2::Preconditioner<>,
              public Ifpack2::Details::CanChangeMatrix<Tpetra::RowMatrix<>>
{
  public:
    //@}
    // \name Constructors and destructors
    //@{

    /// \brief Constructor.
    Hypre();

    //! Destructor
    ~Hypre();

    // @}
    // @{ Construction methods
    //! Initialize the preconditioner, does not touch matrix values.
    void initialize() override;

    //! Returns \c true if the preconditioner has been successfully
    //! initialized.
    bool isInitialized() const { return (is_initialized_); }

    //! Compute ILU factors L and U using the specified graph, diagonal
    //! perturbation thresholds and relaxation parameters.
    /*! This function computes the ILU(k) factors.
     */
    void compute() override;

    //! If factor is completed, this query returns true, otherwise it returns
    //! false.
    bool isComputed() const override { return (is_computed_); }

    //! Set parameters using a Teuchos::ParameterList object. The parameter
    //! list should contain the path the the HypreDrive yaml file.
    void setParameters(const Teuchos::ParameterList& parameterlist) override;

    //@}
    //@{
    /// \brief Set the matrix to be preconditioned.
    ///
    /// \param[in] A The new matrix.
    ///
    /// \post <tt>! isInitialized ()</tt>
    /// \post <tt>! isComputed ()</tt>
    ///
    /// Calling this method resets the preconditioner's state.  After
    /// calling this method with a nonnull input, you must first call
    /// initialize() and compute() (in that order) before you may call
    /// apply().
    ///
    /// You may call this method with a null input.  If A is null, then
    /// you may not call initialize() or compute() until you first call
    /// this method again with a nonnull input.  This method invalidates
    /// any previous factorization whether or not A is null, so calling
    /// setMatrix() with a null input is one way to clear the
    /// preconditioner's state (and free any memory that it may be
    /// using).
    ///
    /// The new matrix A need not necessarily have the same Maps or even
    /// the same communicator as the original matrix.
    void setMatrix(const Teuchos::RCP<const Tpetra::RowMatrix<>>& A);
    //@}

    /// \brief Apply the preconditioner to X, returning the result in Y.
    ///
    /// \param[in] X  A (multi)vector to which to apply the preconditioner.
    /// \param[in,out] Y A (multi)vector containing the result of
    ///   applying the preconditioner to X.
    /// \param[in] mode  If <tt>Teuchos::NO_TRANS</tt>, apply the matrix
    ///   A.  If <tt>mode</tt> is <tt>Teuchos::NO_TRANS</tt>, apply its
    ///   transpose \f$A^T\f$.  If <tt>Teuchos::CONJ_TRANS</tt>, apply
    ///   its Hermitian transpose \f$A^H\f$.
    /// \param[in] alpha  Scaling factor for the result of Chebyshev
    ///   iteration.  The default is 1.
    /// \param[in] beta  Scaling factor for Y.  The default is 0.
    void
    apply(const Tpetra::MultiVector<>& x,
          Tpetra::MultiVector<>& y,
          Teuchos::ETransp mode = Teuchos::NO_TRANS,
          double alpha = Teuchos::ScalarTraits<double>::one(),
          double beta = Teuchos::ScalarTraits<double>::zero()) const override;

    //! The Tpetra::Map representing the domain of this operator.
    Teuchos::RCP<const Tpetra::Map<>> getDomainMap() const override;

    //! The Tpetra::Map representing the range of this operator.
    Teuchos::RCP<const Tpetra::Map<>> getRangeMap() const override;

    //! Whether it's possible to apply the transpose of this operator.
    bool hasTransposeApply() const override;

    //@}
    //! \name Attribute accessor methods
    //@{

    //! The communicator over which the matrix is distributed.
    Teuchos::RCP<const Teuchos::Comm<int>> getComm() const;

    //! The matrix for which this is a preconditioner.
    Teuchos::RCP<const Tpetra::RowMatrix<>> getMatrix() const override;

    //! The number of calls to initialize().
    int getNumInitialize() const override;

    //! The number of calls to compute().
    int getNumCompute() const override;

    //! The number of calls to apply().
    int getNumApply() const override;

    //! The time (in seconds) spent in initialize().
    double getInitializeTime() const override;

    //! The time (in seconds) spent in compute().
    double getComputeTime() const override;

    //! The time (in seconds) spent in apply().
    double getApplyTime() const override;

    //@}
    //! @name Implementation of Teuchos::Describable
    //@{

    //! A simple one-line description of this object.
    std::string description() const override;

    //! Print the object with some verbosity level to a Teuchos::FancyOStream.
    void describe(Teuchos::FancyOStream& out,
                  const Teuchos::EVerbosityLevel verbLevel
                  = Teuchos::Describable::verbLevel_default) const override;

    //@}

  private:
    // @{ Private methods

    //! Copy constructor (use is syntactically forbidden)
    Hypre(const Hypre&);

    //! Assignment operator (use is syntactically forbidded)
    Hypre& operator=(const Hypre&);

    //! Copies matrix data from Tpetra matrix to Hypre matrix.
    int copyTpetraToHypre();

    //! Map generation function
    Teuchos::RCP<const Tpetra::Map<>> makeContiguousColumnMap(
        Teuchos::RCP<const Tpetra::CrsMatrix<>>& Matrix) const;

    //! Destroy
    void destroy();

    // Apply only the hypre operator.
    void
    hypreApply(const Tpetra::MultiVector<>& x, Tpetra::MultiVector<>& y) const;

    // @}
    // @{ Internal data
    //! Pointer to the Tpetra::RowMatrix;
    Teuchos::RCP<const Tpetra::RowMatrix<>> A_;

    //! This objects copy of the ParameterList
    Teuchos::ParameterList List_;

    //! If \c true, initialize() has completed successfully.
    bool is_initialized_;

    //! If \c true, compute() has completed successfully.
    bool is_computed_;

    //! The total number of successful calls to initialize().
    int num_initialize_;

    //! The total number of successful calls to compute().
    int num_compute_;

    /// \brief The total number of successful calls to apply().
    ///
    /// This is "mutable" because apply() is a const method; apply() is
    /// const because it is declared this way in Tpetra::Operator.
    mutable int num_apply_;

    //! The total time in seconds over all calls to initialize().
    double initialize_time_;

    //! The total time in seconds over all calls to compute().
    double compute_time_;

    /// \brief The total time in seconds over all calls to apply().
    ///
    /// This is "mutable" because apply() is a const method; apply() is
    /// const because it is declared this way in Tpetra::Operator.
    mutable double apply_time_;

    //! The Hypre matrix created in initialize()
    mutable HYPRE_IJMatrix hypre_A_;

    //! The Hypre Vector for input
    mutable HYPRE_IJVector hypre_x_;
    mutable HYPRE_ParVector par_x_;

    //! The Hypre Vectors for output
    mutable HYPRE_IJVector hypre_y_;
    mutable HYPRE_ParVector par_y_;

    //! These are linear maps that meet the needs of Hypre
    Teuchos::RCP<const Tpetra::Map<>> globally_contiguous_row_map_;
    Teuchos::RCP<const Tpetra::Map<>> globally_contiguous_col_map_;

    //! YAML input for HypreDrive
    std::string hypredrive_yaml_;

    //! HypreDrive instance.
    HYPREDRV_t hypredrv_;
};

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD

#endif /* VERTEXFCFD_LINEARSOLVERS_HYPRE_HPP */
