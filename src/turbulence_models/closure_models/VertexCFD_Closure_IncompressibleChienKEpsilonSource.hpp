#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONSOURCE_HPP

/**
 * @file IncompressibleChienKEpsilonSource.hpp
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
 * @defgroup ClosureModel Closure Model
 * @brief Turbulence closure models.
 * @{
 */

/**
 * @class IncompressibleChienKEpsilonSource
 * @brief Source term for Chien's low‑Re K‑Epsilon turbulence model
 * (incompressible).
 *
 * This class evaluates the production, destruction, and source terms for the
 * turbulent kinetic energy \f$k\f$ and its dissipation rate \f$\varepsilon\f$
 * using the low‑Re formulation of Chien. It accounts for wall‑distance
 * damping, viscous effects, and a wall‑shear stress correction.
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleChienKEpsilonSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     * @param ir Integration rule providing the data layout.
     * @param global_data Global data object containing problem‑wide values.
     * @param turb_params Parameter list with turbulence model settings,
     *                    e.g. boundary surface area.
     */
    IncompressibleChienKEpsilonSource(
        const panzer::IntegrationRule& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the source terms for the current workset.
     *
     * This method extracts the wall‑shear stress from the global data,
     * computes the wall‑shear stress per unit area, and launches a Kokkos
     * parallel kernel that fills the production, destruction and source
     * fields for \f$k\f$ and \f$\varepsilon\f$.
     *
     * @param workset Evaluation data for the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed for each cell.
     *
     * @param team Team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    //--- Dependent fields -------------------------------------------------
    // Kinematic viscosity \f$\nu\f$
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    // Turbulent eddy viscosity \f$\nu_t\f$
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;
    // Turbulent kinetic energy \f$k\f$
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    // Turbulent dissipation rate \f$\varepsilon\f$
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;
    // Distance to the nearest wall
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;

    // Gradient of the velocity vector \f$\nabla \mathbf{u}\f$
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Global data handle
    Teuchos::RCP<panzer::GlobalData> _global_data;

    //--- Model constants -----------------------------------------------
    double _C_nu;  //!< Model constant \f$C_{\nu}=0.09\f$
    double _C_1;   //!< Model constant \f$C_{1}=1.35\f$
    double _C_2;   //!< Model constant \f$C_{2}=1.8\f$
    double _f_one; //!< Damping factor (set to 1.0)

    // Boundary surface area used to compute wall shear stress per unit area
    double _area;

    // Wall shear stress \f$\tau_w\f$ obtained from global data
    scalar_type _wall_shear_stress;

  public:
    //--- Evaluated fields -----------------------------------------------
    // Source term for the \f$k\f$ equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;
    // Production term for the \f$k\f$ equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;
    // Destruction term for the \f$k\f$ equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;
    // Source term for the \f$\varepsilon\f$ equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_source;
    // Production term for the \f$\varepsilon\f$ equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_prod;
    // Destruction term for the \f$\varepsilon\f$ equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_dest;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONSOURCE_HPP