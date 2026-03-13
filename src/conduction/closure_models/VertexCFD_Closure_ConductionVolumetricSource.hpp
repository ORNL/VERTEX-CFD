#ifndef VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_HPP

/**
 * @file VertexCFD_Closure_ConductionVolumetricSource.hpp
 * @brief Volumetric source/sink evaluator for conduction problems.
 *
 * Provides an evaluator that adds a volumetric heat source (or sink) to the
 * energy equation. The source can be constant or linear in the x‑direction.
 */

/**
 * @defgroup ConductionVolumetricSource Conduction Volumetric Source
 * @brief Evaluators related to conduction source terms.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//----------------------------------------------//
// Volumetric source/sink for conduction.
//----------------------------------------------//

/**
 * @class ConductionVolumetricSource
 * @brief Evaluator that adds a volumetric heat source (or sink) to the
 *        energy equation for conduction problems.
 *
 * The source term can be either a constant value or a linear function of the
 * spatial coordinate in the x‑direction. The specific variation is chosen
 * through the ``Heat Source Type`` input parameter. The evaluator is
 * templated on the evaluation type (e.g. Residual, Jacobian) and the Traits
 * class required by the Panzer/Phalanx infrastructure.
 */
template<class EvalType, class Traits>
class ConductionVolumetricSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /** \brief Constructor.
     *
     *  \param[in] ir            Integration rule that defines the cell
     * topology, number of points, and spatial dimension. \param[in]
     * closure_params Parameter list containing the key
     *                           `"Volumetric Heat Source Value"` which
     * specifies the magnitude of the source \f$ q \f$.
     *
     *  The constructor extracts the source magnitude from `closure_params` and
     *  initializes internal data structures.  No side effects other than
     * member initialization occur.
     */
    ConductionVolumetricSource(const panzer::IntegrationRule& ir,
                               const Teuchos::ParameterList& closure_params);

    /** \brief Register fields with the FieldManager after construction.
     *
     *  This method is called by the Panzer infrastructure during the
     *  post‑registration phase.  It binds the MDField `_source` to the
     *  appropriate data layout and records any dependencies.
     *
     *  \param[in] sd Field manager setup data.
     *  \param[in] fm Reference to the FieldManager.
     */
    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    /** \brief Compute the source term at each integration point.
     *
     *  The source value is either constant (`_q`) or varies linearly with the
     *  x‑coordinate according to the selected `HeatSourceType`.  The result is
     *  stored in the MDField `_source`.
     *
     *  \param[in] workset Evaluation data containing the current workset
     *                     information (cell indices, integration points,
     * etc.).
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /** \brief Kokkos functor for parallel evaluation over teams.
     *
     *  This operator is invoked by a Kokkos `TeamPolicy` to evaluate the
     * source term in parallel.  The implementation mirrors `evaluateFields`
     * but is expressed in a thread‑safe manner.
     *
     *  \param[in] team Team member provided by Kokkos.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /** \brief MDField that holds the evaluated source term.
     *
     *  The field has layout `<Cell,Point>` and type `scalar_type`.  It is
     *  populated during `evaluateFields` (or the Kokkos functor) and can be
     *  accessed by downstream evaluators.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _source;

  private:
    /** \brief Number of spatial dimensions (e.g. 2 or 3). */
    int _num_space_dim;

    /** \brief Degree of the integration rule (cubature order). */
    int _ir_degree;

    /** \brief Magnitude of the volumetric heat source \f$ q \f$. */
    double _q;

    /** \brief Left boundary coordinate in the x‑direction (used for linear
     * source). */
    double _x_left;

    /** \brief Right boundary coordinate in the x‑direction (used for linear
     * source). */
    double _x_right;

    /** \brief Index of the integration rule within the Panzer infrastructure.
     */
    int _ir_index;

    /** \brief Coordinates of the integration points (read‑only). */
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

    /** \brief Enumeration of supported heat source spatial variations. */
    enum HeatSourceType
    {
        /** Constant source independent of space. */
        constant,
        /** Linear variation in the x‑direction. */
        xlinear
    };

    /** \brief Selected heat source type for this instance. */
    HeatSourceType _heat_source_type;
};

//----------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ConductionVolumetricSource

#endif // VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_HPP