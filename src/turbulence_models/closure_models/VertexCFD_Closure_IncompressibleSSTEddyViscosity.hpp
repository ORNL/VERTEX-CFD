#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_HPP

/**
 * @file IncompressibleSST_EddyViscosity.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Models related to turbulence closures.
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
// Turbulent eddy viscosity for Menter's SST K‑Omega turbulence model
//--------------------------------//

/**
 * @class IncompressibleSSTEddyViscosity
 * @brief Incompressible SST turbulent eddy viscosity model.
 *
 * Implements the eddy viscosity calculation for the incompressible
 * SST k‑omega turbulence model. The model uses the turbulent kinetic
 * energy, specific dissipation rate, and their gradients together with
 * model coefficients to compute the eddy viscosity and the SST blending
 * function.
 *
 * @tparam EvalType Evaluation type (e.g., Residual, Jacobian).
 * @tparam Traits   Phalanx traits type.
 * @tparam NumSpaceDim Number of spatial dimensions.
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleSSTEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type associated with the evaluation type. */
    using scalar_type = typename EvalType::ScalarT;
    /** @brief Number of spatial dimensions (compile‑time constant). */
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     *
     * @param ir Integration rule providing cell topology and quadrature.
     * @param turb_params Parameter list containing model coefficients
     *        (beta_star, a_1, sigma_w2).
     * @param on_wall_boundary Flag indicating if the evaluator is used on a
     *        wall boundary (disables distance field).
     */
    IncompressibleSSTEddyViscosity(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& turb_params,
                                   bool on_wall_boundary = false);

    /**
     * @brief Compute eddy viscosity for all cells in the workset.
     *
     * @param workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed by a team of threads.
     *
     * @param team Kokkos team member.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields
    //--------------------------------

    /** Turbulent kinetic energy, \f$k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    /** Specific dissipation rate, \f$\omega\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;
    /** Gradient of turbulent kinetic energy, \f$\nabla k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;
    /** Gradient of specific dissipation rate, \f$\nabla \omega\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;
    /** Distance to the nearest wall. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;
    /** Fluid density, \f$\rho\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    /** Kinematic viscosity, \f$\nu\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    /** Velocity gradient tensor, \f$\partial u_i/\partial x_j\f$. */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Model coefficients
    //----------------------------------------

    /** Coefficient \f$a_1\f$ controlling the blending between the two SST
     * formulations. */
    double _a_1;
    /** Coefficient \f$\beta^*\f$ used in the turbulence length‑scale
     * definition. */
    double _beta_star;
    /** Coefficient \f$\sigma_{\omega 2}\f$ appearing in the cross‑diffusion
     * term. */
    double _sigma_w2;
    /** Flag indicating whether the evaluator is applied on a wall boundary. */
    bool _on_wall_boundary;

    // Temporary variable indices used in shared memory
    // -------------------------

    enum TmpVars
    {
        /// Turbulent kinetic energy (k) temporary.
        TKE,
        /// Specific dissipation rate (ω) temporary.
        OMEGA,
        /// Log‑region related temporary.
        LOG_REGION,
        /// Near‑wall related temporary.
        NEAR_WALL,
        /// Number of temporaries.
        NUM_TMPS
    };

    /** @brief View type for per‑team shared‑memory temporaries. */
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

  public:
    // Evaluated fields
    //--------------------------------

    /** Turbulent eddy viscosity, \f$\nu_t\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
    /** SST blending function, \f$F_1\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sst_blending_function;
};

//---------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_HPP