#ifndef VERTEXCFD_CLOSURE_RADTRANSMUTATIONSOURCEEXACTSOLUTION_HPP
#define VERTEXCFD_CLOSURE_RADTRANSMUTATIONSOURCEEXACTSOLUTION_HPP

/**
 * @file RADTransmutationSourceExactSolution.hpp
 */

#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"
#include "utils/VertexCFD_Utils_Constants.hpp"

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
/**
 * @defgroup ClosureModel Closure Model
 * @brief RAD closure models.
 * @{
 */

/**
 * @class RADTransmutationSourceExactSolution
 * @brief Evaluator that computes the exact solution for the transmutation
 * source term.
 *
 * The class provides the exact solution for the transmutation source term
 * based on the initial species concentrations and time-dependent neutron flux.
 *
 * @tparam EvalType Evaluation type (e.g., Residual, Jacobian).
 * @tparam Traits   Traits class providing evaluation data types.
 */
template<class EvalType, class Traits>
class RADTransmutationSourceExactSolution
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type used for the evaluation (e.g., double or AD type).
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor for RADTransmutationSourceExactSolution.
     *
     * Initializes the evaluator with the given integration rule, species
     * properties, and closure parameters.
     *
     * @param ir              Integration rule providing quadrature points and
     * weights.
     * @param species_prop    Properties of the species involved in the
     * transmutation.
     * @param closure_params  Parameters for configuring the closure model.
     */
    RADTransmutationSourceExactSolution(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop,
        const Teuchos::ParameterList& closure_params);

    /**
     * @brief Evaluate the exact solution fields for all cells in the workset.
     *
     * @param workset Evaluation data containing the number of cells and other
     *                context needed for field evaluation.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed in parallel over cells/teams.
     *
     * Computes the exact solution contributions for each quadrature point.
     *
     * @param team Kokkos team member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_species;

  public:
    // Evaluated fields -------------------------------------------------
    /**
     * @brief Exact solution field for each species.
     */
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _exact_species;
    /**
     * @brief Neutron flux field (optional, based on closure parameters).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _flux;

  private:
    // Dependent fields -------------------------------------------------
    // Constant parameters related to initial condition distribution
    double _a;
    double _b;
    double _kappa;
    double _beta;

    // Parameter for the neutron flux amplitude
    double _flux_amp;
    // Boolean flag to determine whether to calculate the neutron flux
    bool _calc_flux;
    // Taylor series order for the matrix exponential approximation
    int _ts_order;
    // Initial species concentrations
    Kokkos::Array<double, VertexCFD::Constants::MAX_NUM_VIEW> _initial_amp;
    // Exact solution variables
    scalar_type _c1;
    scalar_type _c2;
    scalar_type _c3;
    // Microscopic cross-section and its product with the neutron flux
    // amplitude
    Kokkos::View<double**, Kokkos::LayoutLeft, PHX::mem_space> _mic_cross_section;
    Kokkos::View<double**, Kokkos::LayoutLeft, PHX::mem_space> _exp_A_p;
    // Time variable for the evaluation
    double _time;
};

//---------------------------------------------------------------------------//

} // namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_RADTRANSMUTATIONSOURCEEXACTSOLUTION_HPP
