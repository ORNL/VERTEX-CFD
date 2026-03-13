#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_HPP

/**
 * @file IncompressibleChienKEpsilonEddyViscosity.hpp
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
 * @class IncompressibleChienKEpsilonEddyViscosity
 * @brief Implements the turbulent eddy viscosity closure for the
 * low‑Reynolds‑number Chien K‑Epsilon model.
 *
 * The model computes the eddy viscosity \f$\nu_t\f$ using the turbulent
 * kinetic energy \f$k\f$, the dissipation rate \f$\varepsilon\f$, the distance
 * to the nearest wall, and the molecular viscosity \f$\nu\f$. The formulation
 * follows Chien's low‑Re model with empirical constants \f$C_{\nu}=0.09\f$ and
 * \f$C_{\tau}=-0.0115\f$.
 *
 * @tparam EvalType Evaluation type.
 * @tparam Traits   Traits type.
 */
template<class EvalType, class Traits>
class IncompressibleChienKEpsilonEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     *
     * Registers dependent and evaluated fields and stores model parameters.
     *
     * @param ir          Integration rule providing cell topology and
     * quadrature.
     * @param global_data Global data object for accessing the parameter list.
     * @param turb_params Parameter list containing turbulence model settings,
     *                    e.g. "Boundary Surface Area".
     */
    IncompressibleChienKEpsilonEddyViscosity(
        const panzer::IntegrationRule& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the eddy viscosity field for the current workset.
     *
     * Retrieves the wall shear stress from the global parameter list,
     * launches a Kokkos parallel kernel, and stores the result in \c _nu_t.
     *
     * @param workset Evaluation data containing cell and point information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed for each cell.
     *
     * Computes \f$\nu_t\f$ at each quadrature point using the smooth
     * maximum function to avoid division by zero.
     *
     * @param team Kokkos team member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields
    // ---------------------------------------------------------

    /**
     * @brief Turbulent kinetic energy \f$k\f$ (scalar) at each cell and point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /**
     * @brief Turbulent dissipation rate \f$\varepsilon\f$ (scalar) at each
     * cell and point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;

    /**
     * @brief Molecular (kinematic) viscosity \f$\nu\f$ at each cell and point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    /**
     * @brief Distance to the nearest wall at each cell and point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;

    // Global data
    // ---------------------------------------------------------

    /**
     * @brief Global data pointer used to access the parameter list.
     */
    Teuchos::RCP<panzer::GlobalData> _global_data;

    // Model constants
    // ---------------------------------------------------------

    /**
     * @brief Empirical constant \f$C_{\nu}=0.09\f$.
     */
    double _C_nu;

    /**
     * @brief Empirical constant \f$C_{\tau}=-0.0115\f$.
     */
    double _C_tau;

    /**
     * @brief Number of spatial dimensions (gradient dimension).
     */
    int _num_grad_dim;

    /**
     * @brief Boundary surface area used to scale wall shear stress.
     */
    double _area;

    /**
     * @brief Wall shear stress (computed from global parameters) divided by
     * area.
     */
    scalar_type _wall_shear_stress;

  public:
    // Evaluated field
    // ---------------------------------------------------------

    /**
     * @brief Turbulent eddy viscosity \f$\nu_t\f$ (output) at each cell and
     * point.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_HPP