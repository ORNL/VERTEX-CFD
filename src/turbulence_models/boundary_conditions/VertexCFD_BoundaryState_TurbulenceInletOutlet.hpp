#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_HPP

/**
 * @file TurbulenceInletOutlet.hpp
 */

/**
 * @defgroup BoundaryCondition Boundary Condition
 * @brief Module for boundary condition implementations.
 * @{
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
//--------------------------------//
/**
 * @class TurbulenceInletOutlet
 * @brief Implements an inlet/outlet boundary condition for a turbulence scalar
 * quantity (e.g. turbulent kinetic energy).
 *
 * The condition blends a prescribed inlet value with the interior solution
 * based on the sign of the normal velocity. A smooth ramp function is used to
 * avoid discontinuities at the inlet/outlet interface.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class required by Phalanx.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g. 2 or 3).
 */
//--------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class TurbulenceInletOutlet : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    // Type alias for the scalar type associated with the evaluation.
    using scalar_type = typename EvalType::ScalarT;
    // Compile‑time constant for the spatial dimension.
    static constexpr int num_space_dim = NumSpaceDim;

    //================================//
    /**
     * @brief Constructor.
     *
     * Registers evaluated and dependent fields with the Phalanx infrastructure
     * and extracts the inlet value from the parameter list.
     *
     * @param ir Integration rule defining the evaluation points and data
     * layout.
     * @param bc_params Parameter list containing the inlet value (key:
     * "<variable_name> Inlet Value").
     * @param variable_name Name of the turbulence variable (e.g. "k").
     */
    //================================//
    TurbulenceInletOutlet(const panzer::IntegrationRule& ir,
                          const Teuchos::ParameterList& bc_params,
                          const std::string variable_name);

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
     * For each cell and integration point, computes the dot product \f$
     * \vec{v}\cdot\vec{n} \f$, applies a smooth ramp to decide inlet vs.
     * outlet, and assigns the appropriate boundary value and gradient.
     */
    //================================//
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    // Evaluated field containing the boundary value of the turbulence
    // variable.
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_variable;

    // Evaluated field containing the gradient of the boundary variable.
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_variable;

  private:
    // Dependent field: interior value of the turbulence variable.
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _variable;
    // Dependent field: interior gradient of the turbulence variable.
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;
    // Dependent field: outward normal vectors at the boundary.
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    // Dependent vector field: velocity components at the boundary.
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    // Prescribed inlet value for the turbulence variable.
    double _inlet_value;
};

//--------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

/** @} */ // end of BoundaryCondition

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_HPP