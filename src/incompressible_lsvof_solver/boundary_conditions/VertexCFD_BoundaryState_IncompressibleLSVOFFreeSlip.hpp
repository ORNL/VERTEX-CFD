#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFFREESLIP_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFFREESLIP_HPP

/**
 * @file IncompressibleLSVOFFreeSlip.hpp
 */

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//--------------------------------//
/**
 * @class IncompressibleLSVOFFreeSlip
 * @brief Implements an free slip (symmetry) condition for the LSVOF DOFs as
 * well as velocity and pressure.
 *
 * The condition prescribes a no penetration condition for the velocity, and
 * zero gradient normal to the wall for all gradients.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class required by Phalanx.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g. 2 or 3).
 */
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFFreeSlip
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    // Type alias for the scalar type associated with the evaluation.
    using scalar_type = typename EvalType::ScalarT;
    // Compile-time constant for the spatial dimension.
    static constexpr int num_space_dim = NumSpaceDim;

    //================================//
    /**
     * @brief Constructor.
     *
     * Registers evaluated and dependent fields with the Phalanx infrastructure
     * and extracts the LSVOF model type from the parameter list.
     *
     * @param ir Integration rule defining the evaluation points and data
     * layout.
     * @param num_lsvof_dofs Number of LSVOF variables to solve for; generally
     * equal to the number of phases minus 1.
     * @param lsvof_model_name Name of the LSVOF model in use
     * @param continuity_model_name Name of the continuity model (i.e., AC or
     * EDAC).
     * @param build_mom_equ Bool to control whether or not the momentum and
     * continuity equations are in use.
     */
    //================================//
    IncompressibleLSVOFFreeSlip(const panzer::IntegrationRule& ir,
                                const int& num_lsvof_dofs,
                                const std::string& lsvof_model_name,
                                const std::string& continuity_model_name,
                                const bool& build_mom_equ);

    //================================//
    /**
     * @brief Evaluate the boundary fields for the current workset.
     *
     * Launches a Kokkos team‑parallel kernel that invokes the functor
     * operator() for each cell.
     */
    //================================//
    void evaluateFields(typename Traits::EvalData workset) override;

    //================================//
    /**
     * @brief Functor operator executed by Kokkos.
     *
     * For each cell and integration point on the boundary, subtracts the
     * normal component of the velocity, and the normal components of all
     * gradients, from the interior value, and sets these zero-normal gradients
     * as the boundary value.
     */
    //================================//
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Phalanx data layout for VOF phase field vector
    Teuchos::RCP<PHX::DataLayout> _phase_layout;

  public:
    // Evaluated field containing the boundary value of the pressure field
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _boundary_lagrange_pressure;

    // Evaluated field containing the boundary pressure gradient
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_lagrange_pressure;

    // Evaluated field containing the boundary velocity field
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _boundary_velocity;

    // Evaluated field containing the boundary velocity gradient
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _boundary_grad_velocity;

    // Evaluated field containing the boundary values of the VOF variables
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _boundary_alphas;

    // Evaluated field containing the boundary level set field
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_phi;

  private:
    // Dependent field: interior pressure field
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;

    // Dependent field: interior pressure gradient
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    // Dependent field: interior velocity
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    // Dependent field: interior velocity gradient
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Dependent field: outward-facing boundary normal vector
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;

    // Dependent field: interior VOF variables
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _alphas;

    // Dependent field: interior level set field
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _phi;

    // Enum containing the LSVOF model options
    enum LSVOFModelType
    {
        VOF,
        CLS
    };

    // LSVOF model type to be evaluated
    LSVOFModelType _lsvof_model_type;

    // Bool controlling whether AC or EDAC model is used
    bool _is_edac;

    // Bool controlling whether NS equations are solved
    bool _build_mom_equ;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFFREESLIP_HPP
