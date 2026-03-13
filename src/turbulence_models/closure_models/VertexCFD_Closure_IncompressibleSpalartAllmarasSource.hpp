#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_HPP

/**
 * @file IncompressibleSpalartAllmarasSource.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Closure models for turbulence.
 *
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
//--------------------------------//
// Source term for Spalart-Allmaras turbulence model (SA-neg)
//--------------------------------//

/**
 * @class IncompressibleSpalartAllmarasSource
 * @brief Source term for Spalart‑Allmaras turbulence model (SA‑neg).
 *
 * Evaluates the source term for each cell/point in the workset using the
 * Spalart‑Allmaras formulation.
 *
 * @tparam EvalType Evaluation type providing scalar type definition.
 * @tparam Traits   Traits class defining evaluation data.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleSpalartAllmarasSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /**
     * @brief Alias for the scalar type used in the evaluation.
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Number of spatial dimensions (compile‑time constant).
     */
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor that registers dependent and evaluated fields.
     *
     * @param ir Integration rule providing data layout for fields.
     */
    IncompressibleSpalartAllmarasSource(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the source term for each cell/point in the workset.
     *
     * @param workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Functor executed by Kokkos parallel_for over cells.
     *
     * @param team Team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /**
     * @var _sa_var
     * @brief Spalart‑Allmaras working variable (e.g., turbulent viscosity).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sa_var;

    /**
     * @var _distance
     * @brief Wall distance field used for damping functions.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;

    /**
     * @var _grad_sa_var
     * @brief Gradient of the SA variable.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_sa_var;

    /**
     * @var _nu
     * @brief Kinematic viscosity of the fluid.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    /**
     * @var _grad_velocity
     * @brief Gradient of the velocity field (vector of size NumSpaceDim).
     */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Model constants (see Spalart‑Allmaras formulation).

    /**
     * @brief Turbulent Prandtl number.
     */
    const double _sigma;

    /**
     * @brief von Kármán constant.
     */
    const double _kappa;

    /**
     * @brief Production constant.
     */
    const double _c_b1;

    /**
     * @brief Diffusion constant.
     */
    const double _c_b2;

    /**
     * @brief Transition constant.
     */
    const double _c_t3;

    /**
     * @brief Transition constant.
     */
    const double _c_t4;

    /**
     * @brief Viscosity function constant.
     */
    const double _c_v1;

    /**
     * @brief Viscosity function constant.
     */
    const double _c_v2;

    /**
     * @brief Viscosity function constant.
     */
    const double _c_v3;

    /**
     * @brief Destruction constant.
     */
    const double _c_w1;

    /**
     * @brief Destruction constant.
     */
    const double _c_w2;

    /**
     * @brief Destruction constant.
     */
    const double _c_w3;

    /**
     * @brief Upper limit for the \f$r\f$‑function.
     */
    const scalar_type _rlim;

  public:
    /**
     * @var _sa_source
     * @brief Computed source term (production + destruction + diffusion).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sa_source;

    /**
     * @var _sa_prod
     * @brief Production component of the SA source term.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sa_prod;

    /**
     * @var _sa_dest
     * @brief Destruction component of the SA source term.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sa_dest;
};

//--------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_HPP