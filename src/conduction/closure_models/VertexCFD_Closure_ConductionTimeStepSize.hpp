#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_HPP

/**
 * @file VertexCFD_Closure_ConductionTimeStepSize.hpp
 * @brief Compute a stable time‑step size for the conduction equation.
 *
 * This header defines the @c ConductionTimeStepSize evaluator which calculates
 * a local diffusion CFL‑based time‑step for each cell and integration point.
 */

/**
 * @defgroup ConductionTimeStepSize Conduction Time‑Step Size
 * @brief Module providing the conduction time‑step size evaluator.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace ClosureModel
{

/// \brief Compute a stable time‑step size for the conduction equation.
///
/// The time‑step is based on a local diffusion CFL condition using the
/// thermal conductivity, specific heat, solid density and element length.
///
/// The evaluator produces a scalar field \c local_dt defined at each cell and
/// integration point.
template<class EvalType, class Traits>
class ConductionTimeStepSize : public panzer::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type used by the evaluation type.
     *
     *  Typically this is either @c double or a Sacado
     * automatic‑differentiation type.
     *
     *  @tparam EvalType Evaluation type providing the scalar definition.
     *  @tparam Traits   Traits class used by the Phalanx framework.
     */
    using scalar_type = typename EvalType::ScalarT;

    /** @brief Constructor.
     *
     *  Registers the fields that this evaluator computes and depends on.
     *
     *  @param[in] ir Integration rule providing data layout information.
     */
    ConductionTimeStepSize(const panzer::IntegrationRule& ir);

    /** @brief Evaluate the time‑step size on the supplied workset.
     *
     *  This method is called by the Phalanx evaluation engine.  It launches a
     *  Kokkos team‑level parallel loop that invokes the functor @c operator().
     *
     *  @param[in] workset Evaluation data containing the cell range.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /** @brief Functor invoked by Kokkos for each team of cells.
     *
     *  Computes the local diffusion time‑step size for each cell and
     * integration point using the formula \f[ \Delta t = \frac{\rho c_p \,
     * \Delta x^2}{2 k}, \f] where \f$\rho\f$ is solid density, \f$c_p\f$ is
     * specific heat, \f$\Delta x\f$ is the element length, and \f$k\f$ is
     * thermal conductivity.
     *
     *  @param[in] team Kokkos team member representing a subset of cells.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /** @brief Output field: local time‑step size.
     *
     *  Dimensions: (Cell, Point)
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _local_dt;

  private:
    /** @brief Input field: thermal conductivity (k) at each cell/point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _thermal_conductivity;

    /** @brief Input field: specific heat capacity (c_p) at each cell/point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _specific_heat;

    /** @brief Input field: solid density (ρ) at each cell/point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _solid_density;

    /** @brief Input field: element length vector (Δx) at each cell/point. */
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;
};

//@}

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_HPP