#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP

/**
 * @file IncompressibleWALEEddyViscosity.hpp
 */

/**
 * @defgroup ClosureModel Incompressible WALE Eddy Viscosity
 * @brief Implementation of the WALE sub‑grid model for incompressible flows.
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
//--------------------------------
// Turbulent eddy viscosity for WALE LES turbulence model
//-------------------------------------------------------

/**
 * @class IncompressibleWALEEddyViscosity
 * @brief Incompressible WALE (Wall‑Adapting Local Eddy‑viscosity) sub‑grid
 * model.
 *
 * This evaluator computes the sub‑grid kinetic energy \f$k_{sgs}\f$ and the
 * corresponding eddy viscosity \f$\nu_t\f$ for an incompressible flow using
 * the WALE LES formulation. The model requires the element length scale and
 * the gradient of the velocity field as inputs.
 *
 * @tparam EvalType Type providing the scalar type (e.g., residual, Jacobian).
 * @tparam Traits   Traits class defining the evaluation data type.
 * @tparam NumSpaceDim Spatial dimension (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleWALEEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     *
     * Registers dependent fields (element length and velocity gradient) and
     * evaluated fields (sub‑grid kinetic energy and eddy viscosity). Model
     * coefficients \f$C_k\f$ and \f$C_w\f$ are taken from the supplied
     * parameter list, falling back to default values if not provided.
     *
     * @param ir          Integration rule providing field layout information.
     * @param turb_params Parameter list containing optional coefficients
     *                    "C_k" and "C_w".
     */
    IncompressibleWALEEddyViscosity(const panzer::IntegrationRule& ir,
                                    const Teuchos::ParameterList& turb_params);

    /**
     * @brief Compute the eddy viscosity and sub‑grid kinetic energy.
     *
     * This method launches a Kokkos team‑parallel kernel that evaluates the
     * WALE model at each quadrature point.
     *
     * @param workset Evaluation data containing the cell range.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed by each team.
     *
     * Implements the WALE formulation for a single cell. The implementation
     * resides in the corresponding *_impl.hpp file.
     *
     * @param team Kokkos team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /**
     * @brief Dependent field: element length scale \f$\Delta\f$ for each
     * spatial direction.
     */
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;

    /**
     * @brief Dependent field: gradient of velocity components.
     */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    /**
     * @brief Coefficient for sub‑grid kinetic energy (default 0.094).
     */
    double _C_k;
    /**
     * @brief Coefficient for eddy viscosity (default 0.275).
     */
    double _C_w;

    /**
     * @brief Temporary variable indices used in shared memory scratch view.
     */
    enum TmpVars
    {
        /// \f$|S|^2\f$ – magnitude squared of the symmetric part.
        MAG_SQR_S,
        /// \f$|W|^2\f$ – magnitude squared of the skew‑symmetric part.
        MAG_SQR_W,
        /// \f$|S_d|^2\f$ – magnitude squared of the traceless symmetric part.
        MAG_SQR_SD,
        /// Temporary storage for individual components of \f$S_d\f$.
        SD_IJ,
        /// Number of temporaries.
        NUM_TMPS
    };

    /**
     * @brief View type for per‑team shared memory temporaries.
     */
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

  public:
    // Evaluated fields.
    /**
     * @brief Sub‑grid kinetic energy \f$k_{sgs}\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_sgs;
    /**
     * @brief Eddy viscosity \f$\nu_t\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//-----------------------------------------------//

} // end namespace ClosureModel
/** @} */ // end of ClosureModel group
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_HPP