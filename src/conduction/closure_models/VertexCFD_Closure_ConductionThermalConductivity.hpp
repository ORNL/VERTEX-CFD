#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_HPP

/**
 * @file VertexCFD_Closure_ConductionThermalConductivity.hpp
 * @brief Evaluates temperature‑dependent thermal conductivity for conduction
 * models.
 *
 * Provides the ConductionThermalConductivity evaluator used in closure models.
 */

/**
 * @defgroup VertexCFD_Closure_ConductionThermalConductivity Conduction Thermal
 * Conductivity Closure
 * @brief Closure model for thermal conductivity in conduction simulations.
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

/**
 * @class ConductionThermalConductivity
 * @brief Evaluates temperature‑dependent thermal conductivity for conduction
 * models.
 *
 * The computed field @c thermal_conductivity follows one of two functional
 * forms:
 * - Constant conductivity: \f$ k(T) = k_0 \f$
 * - Inverse‑proportional conductivity: \f$ k(T) = \dfrac{k_0}{T} \f$
 *
 * where \f$ k_0 \f$ is the base conductivity coefficient and \f$ T \f$ is the
 * temperature.
 *
 * @tparam EvalType Evaluation type (e.g., Residual, Jacobian).
 * @tparam Traits  Traits class providing evaluation data types.
 */
template<class EvalType, class Traits>
class ConductionThermalConductivity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type used by the evaluator (extracted from @c EvalType).
     */
    using scalar_type = typename EvalType::ScalarT;

    /** @brief Constructor.
     *
     *  Registers the evaluated field and reads closure parameters that
     *  define the thermal‑conductivity model.
     *
     *  @param[in] ir               Integration rule providing the data layout.
     *  @param[in] closure_params   Parameter list containing the model
     *                              type ("Thermal Conductivity Type")
     *                              and associated coefficients.
     */
    ConductionThermalConductivity(const panzer::IntegrationRule& ir,
                                  const Teuchos::ParameterList& closure_params);

    /** @brief Compute the thermal conductivity field for the current workset.
     *
     *  This method is invoked by the Phalanx evaluation engine.
     *
     *  @param[in] workset  Evaluation data (cell indices, etc.).
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /** @brief Kokkos functor for parallel evaluation over a team policy.
     *
     *  The implementation loops over cells and integration points,
     *  applying the selected conductivity model.
     *
     *  @param[in] team  Team member provided by Kokkos.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /** @brief Evaluated field containing the computed thermal conductivity.
     *
     *  Indexed as <tt>_thermal_conductivity(cell,point)</tt>.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _thermal_conductivity;

  private:
    /** @brief Dependent field: temperature at each cell and point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;

    /** @brief Number of spatial dimensions (e.g., 2 or 3). */
    int _num_space_dim;

    /** @brief Base conductivity coefficient (k0) read from the parameter list.
     */
    double _k0;

    /** @brief Supported thermal‑conductivity functional forms. */
    enum ThermCondType
    {
        /** @brief Constant conductivity (k = k0). */
        constant,
        /** @brief Inverse‑proportional conductivity (k = k0 / T). */
        one_over_temp
    };

    /** @brief Selected conductivity model for this instance. */
    ThermCondType _therm_cond_type;
};

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of VertexCFD_Closure_ConductionThermalConductivity

#endif // VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_HPP