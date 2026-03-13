#ifndef VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_HPP

/**
 * @file VertexCFD_Closure_ConductionExactSolution.hpp
 * @brief Exact analytical solution evaluator for steady conduction.
 *
 * Provides an evaluator that registers and computes the exact pressure
 * (Lagrange multiplier), velocity, and temperature fields for a prescribed
 * analytical solution of a steady conduction problem.
 */

/**
 * @defgroup VertexCFD_Closure_ConductionExactSolution Conduction Exact
 * Solution
 * @brief Doxygen module for the conduction exact‑solution evaluator.
 *
 * This module contains the @c ConductionExactSolution class which evaluates
 * the analytical solution fields based on problem parameters.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{

/** \brief Evaluator providing the exact analytical solution for a
 *         steady conduction problem.
 *
 *  This evaluator registers the exact pressure (Lagrange multiplier),
 *  velocity, and temperature fields for a prescribed analytical solution.
 *  The solution depends on a constant volumetric heat source \f$ q \f$,
 *  a thermal conductivity \f$ k \f$, and a prescribed temperature on the
 *  right boundary \f$ T_{\text{right}} \f$.  These parameters are read
 *  from the supplied \c closure_params list.
 *
 *  The evaluator is templated on the evaluation type \c EvalType,
 *  the traits class \c Traits, and the spatial dimension \c NumSpaceDim.
 *
 * @tparam EvalType   Evaluation type providing the scalar type.
 * @tparam Traits     Traits class used by the evaluator framework.
 * @tparam NumSpaceDim Compile‑time number of spatial dimensions.
 */
template<class EvalType, class Traits, int NumSpaceDim>
class ConductionExactSolution : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** \brief Scalar type used by the evaluation type. */
    using scalar_type = typename EvalType::ScalarT;
    /** \brief Number of spatial dimensions (compile‑time constant). */
    static constexpr int num_space_dim = NumSpaceDim;

    /** \brief Constructor.
     *
     *  Registers the evaluated fields and extracts the exact‑solution
     *  parameters from the closure parameter list.
     *
     *  \param[in] ir Integration rule providing cell topology and data layout
     * objects. \param[in] closure_params Parameter list containing:
     *      - "Volumetric Heat Source Value"
     *      - "Thermal Conductivity Coefficient"
     *      - "Right Temperature Boundary Value"
     */
    ConductionExactSolution(const panzer::IntegrationRule& ir,
                            const Teuchos::ParameterList& closure_params);

    /** \brief Perform post‑registration setup.
     *
     *  Retrieves integration‑point coordinates and stores the integration
     *  rule degree for later use during field evaluation.
     *
     *  \param[in] sd   Setup data provided by the Traits class.
     *  \param[in] fm   Field manager (unused, required by interface).
     */
    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    /** \brief Evaluate the exact solution fields on the workset.
     *
     *  Computes the pressure, velocity, and temperature at each integration
     *  point using the analytical expressions defined by the model
     *  parameters \f$ q, k, T_{\text{right}} \f$.
     *
     *  \param[in] workset  Evaluation data containing cell and point
     *                      information for the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /** \brief Kokkos functor for team‑level parallel evaluation.
     *
     *  This operator is invoked by a Kokkos \c TeamPolicy to evaluate the
     *  exact solution for a team of cells.  The implementation mirrors
     *  \c evaluateFields but operates on the Kokkos team abstraction.
     *
     *  \param[in] team  Kokkos team member representing a group of cells.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /** \brief Exact Lagrange pressure field.
     *
     *  Stored as a scalar field indexed by cell and integration point.
     */
    PHX::MDField<double, panzer::Cell, panzer::Point> _lagrange_pressure;

    /** \brief Exact velocity vector field.
     *
     *  One component per spatial dimension, each stored as a scalar field
     *  indexed by cell and integration point.
     */
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        _velocity;

    /** \brief Exact temperature field.
     *
     *  Stored as a scalar field indexed by cell and integration point.
     */
    PHX::MDField<double, panzer::Cell, panzer::Point> _temperature;

  private:
    /** \brief Degree of the integration rule used for this evaluator. */
    int _ir_degree;

    /** \brief Volumetric heat source value \f$ q \f$. */
    double _q;

    /** \brief Thermal conductivity coefficient \f$ k \f$. */
    double _k;

    /** \brief Prescribed temperature on the right boundary \f$
     * T_{\text{right}} \f$. */
    double _T_right;

    /** \brief Index of the integration rule degree within the workset. */
    int _ir_index;

    /** \brief Integration‑point coordinates (x, y, z) for each cell. */
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
};

/// @} // end of VertexCFD_Closure_ConductionExactSolution group

} // namespace ClosureModel
} // namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_HPP