#ifndef VERTEXCFD_CLOSURE_PLASMASPECIESTIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_PLASMASPECIESTIMEDERIVATIVE_HPP

#include "utils/VertexCFD_Utils_VelocityDim.hpp"

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
//---------------------------------------------------------------------------//
/**
 * @brief Compute the time derivatives of the conserved variables for the
 * species equations (number density equation, momentun density equation,
 * and energy density equation).
 *
 * @tparam EvalType Evaluation type (Residual, Jacobian, Tangent, Hessian).
 *
 * @tparam Traits Panzer traits for types (e.g. panzer::Traits).
 *
 * @tparam NumSpaceDim Dimension of the computational domain (2 or 3).

 * @see tstPlasmaSpeciesTimeDerivative.cpp for examples.
*/
template<class EvalType, class Traits, int NumSpaceDim>
class PlasmaSpeciesTimeDerivative
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor
     *
     * @param ir Integration rule for constructing the quadrature point
     * evaluator.
     *
     * @param species_name name of the species to compute the time derivative
     * for.
     */
    PlasmaSpeciesTimeDerivative(const panzer::IntegrationRule& ir,
                                const std::string species_name);

    /** \brief Evaluate the time derivatives of the conservative variables on
     * the workset.
     *
     *  \param[in] workset  Evaluation data containing cell and point
     *                      information for the current workset.
     */
    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /// Time derivative for number density equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_continuity;

    /// Time derivative for momentum density equation
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _dqdt_momentum;

    // Time derivative for energy density equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_energy;

  private:
    // Species velocity
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, VelocityDim>
        _velocity;
    // Species density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;

    /// Time derivative of the species number density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_nd;
    /// Time derivative of the species velocity
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, VelocityDim>
        _dxdt_velocity;
    /// Time derivative of the species density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dvdt_rho;
    /// Time derivative of the species energy density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dvdt_E;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_PLASMASPECIESTIMEDERIVATIVE_HPP
