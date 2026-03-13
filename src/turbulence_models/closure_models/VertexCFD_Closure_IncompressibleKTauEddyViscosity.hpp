#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUEDDYVISCOSITY_HPP

/**
 * @file IncompressibleKTauEddyViscosity.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Models that provide closure relations for turbulence.
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

/**
 * @class IncompressibleKTauEddyViscosity
 * @brief Closure model that computes the turbulent eddy viscosity for the
 * incompressible K‑Tau turbulence model.
 *
 * @tparam EvalType Evaluation type (e.g., Residual, Jacobian) providing the
 * scalar type used for field values.
 * @tparam Traits   Phalanx traits class.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKTauEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type associated with the evaluation type. */
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor that registers dependent and evaluated fields.
     * @param ir Integration rule providing the data layout for fields.
     */
    IncompressibleKTauEddyViscosity(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the eddy viscosity at each quadrature point.
     * @param workset Evaluation data containing cell and point information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed per team (cell) to compute the viscosity.
     * @param team Team member representing a cell in the parallel launch.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /** @brief Input field: turbulent kinetic energy \f$k\f$, defined at each
     * cell and quadrature point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /** @brief Input field: specific dissipation rate \f$\omega\f$, defined at
     * each cell and quadrature point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

  public:
    /** @brief Output field: turbulent eddy viscosity \f$\nu_t = k \,
     * \omega\f$, evaluated at each cell and quadrature point. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUEDDYVISCOSITY_HPP