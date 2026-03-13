#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONSOURCE_HPP

/**
 * @file IncompressibleRealizableKEpsilonSource.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Turbulence closure models.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//--------------------------------------------//
// Source term for realizable K‑Epsilon turbulence model
//--------------------------------------------//

/**
 * @class IncompressibleRealizableKEpsilonSource
 * @brief Evaluates the source, production, and destruction terms for the
 *        incompressible realizable k‑ε turbulence model.
 *
 * @tparam EvalType Evaluation type that defines the scalar type.
 * @tparam Traits   Traits class required by the Phalanx evaluator framework.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleRealizableKEpsilonSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor that registers dependent and evaluated fields.
     * @param ir Integration rule providing the data layout for fields.
     */
    IncompressibleRealizableKEpsilonSource(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the source terms for each cell in the workset.
     * @param workset Evaluation data containing cell and point information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed in parallel over cells.
     * @param team Team policy member representing a cell (team).
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields ------------------------------------------//

    /** @brief Kinematic viscosity, \f$\nu\f$, at each integration point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    /** @brief Turbulent eddy viscosity, \f$\nu_t\f$, at each integration
     * point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    /** @brief Turbulent kinetic energy, \f$k\f$, at each integration point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /** @brief Turbulent dissipation rate, \f$\varepsilon\f$, at each
     * integration point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;

    /**
     * @brief Gradient of the velocity vector, \f$\nabla u\f$, stored as a
     * rank‑4 field: (component \f$i\f$, cell, point, spatial direction
     * \f$j\f$).
     */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Model constant -------------------------------------------//

    /** @brief Model constant \f$C_2\f$ (default value 1.9) used in the
     * \f$\varepsilon\f$ destruction term. */
    double _C_2;

  public:
    // Evaluated fields ------------------------------------------//

    /** @brief Total source term for the turbulent kinetic‑energy equation
     * (\f$k\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;

    /** @brief Production term for the turbulent kinetic‑energy equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;

    /** @brief Destruction (dissipation) term for the turbulent kinetic‑energy
     * equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;

    /** @brief Total source term for the turbulent dissipation‑rate
     * (\f$\varepsilon\f$) equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_source;

    /** @brief Production term for the turbulent dissipation‑rate equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_prod;

    /** @brief Destruction term for the turbulent dissipation‑rate equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_dest;
};

//-----------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONSOURCE_HPP