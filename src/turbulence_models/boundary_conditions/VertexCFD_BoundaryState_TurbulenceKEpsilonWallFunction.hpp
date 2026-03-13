#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_HPP

/**
 * @file TurbulenceKEpsilonWallFunction.hpp
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{

/**
 * @class TurbulenceKEpsilonWallFunction
 * @brief Implements a wall‑function boundary condition for the
 * \f$k\f$–\f$\varepsilon\f$ turbulence model.
 *
 * @tparam EvalType  Evaluation type (e.g. Residual, Jacobian).
 * @tparam Traits    Traits class providing the evaluation data type.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g. 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class TurbulenceKEpsilonWallFunction
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     *
     * Initializes field tags and reads user‑specified options.
     *
     * @param ir         Integration rule providing cell topology and
     * quadrature information.
     * @param bc_params  Parameter list containing boundary‑condition
     * specifications (e.g. "Epsilon Condition Type").
     */
    TurbulenceKEpsilonWallFunction(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& bc_params);

    /**
     * @brief Evaluate the boundary fields for the given workset.
     *
     * This method launches a Kokkos parallel kernel that computes the
     * wall‑function values at each quadrature point.
     *
     * @param workset  Evaluation data containing cell and point indices.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Functor executed by Kokkos for each cell.
     *
     * Computes turbulent kinetic energy, dissipation rate, friction
     * velocity, and related quantities on the wall.
     *
     * @param team  Kokkos team member representing a single cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /**
     * @brief Boundary turbulent kinetic energy \f$\tilde{k}\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_k;

    /**
     * @brief Boundary turbulent dissipation rate \f$\tilde{\varepsilon}\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_e;

    /**
     * @brief Gradient of boundary turbulent kinetic energy
     *        \f$\partial \tilde{k} / \partial x_i\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_k;

    /**
     * @brief Gradient of boundary turbulent dissipation rate
     *        \f$\partial \tilde{\varepsilon} / \partial x_i\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_e;

    /**
     * @brief Boundary friction velocity \f$u_\tau\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_u_tau;

    /**
     * @brief Non‑dimensional wall distance \f$y^+\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_y_plus;

    /**
     * @brief Turbulent eddy viscosity computed from the wall function.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _wall_func_nu_t;

  private:
    /**
     * @brief Interior turbulent kinetic energy field (input).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _k;

    /**
     * @brief Interior turbulent dissipation rate field (input).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _e;

    /**
     * @brief Velocity components \f$\mathbf{u}\f$ at the boundary.
     */
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    /**
     * @brief Gradient of interior turbulent kinetic energy.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_k;

    /**
     * @brief Gradient of interior turbulent dissipation rate.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_e;

    /**
     * @brief Outward unit normal vectors at the boundary.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;

    /**
     * @brief Kinematic viscosity \f$\nu\f$ (input).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    /**
     * @brief Number of gradient dimensions (derived from the mesh).
     */
    int _num_grad_dim;

    /**
     * @brief Model constant \f$C_\mu = 0.09\f$.
     */
    double _C_mu;

    /**
     * @brief von Kármán constant \f$\kappa = 0.41\f$.
     */
    double _kappa;

    /**
     * @brief Transition value for \f$y^+\f$ (typically \f$y^+_{\text{tr}}
     * = 11.06\f$).
     */
    double _yp_tr;

    /**
     * @brief Flag indicating whether a Neumann condition is applied to
     * \f$\varepsilon\f$.
     */
    bool _neumann;
};

} // namespace BoundaryCondition
} // namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_HPP