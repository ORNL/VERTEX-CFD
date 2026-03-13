#ifndef VERTEXCFD_CLOSURE_PLASMASPECIESEOS_HPP
#define VERTEXCFD_CLOSURE_PLASMASPECIESEOS_HPP

#include "utils/VertexCFD_Utils_VelocityDim.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
/**
 * @brief Compute the pressure, the energy density, and the momentum density
 from the primitive variables and an equation of state. Time derivative of the
 energy density and the momentum density are evaluated using the chain rule.
 *
 * @tparam EvalType Evaluation type (Residual, Jacobian, Tangent, Hessian).
 *
 * @tparam Traits Panzer traits for types (e.g. panzer::Traits).
 *

 * @see tstPlasmaSpeciesEOS.cpp for examples.
*/
template<class EvalType, class Traits>
class PlasmaSpeciesEOS : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor
     *
     * @param ir Integration rule for constructing the quadrature point
     * evaluator.
     *
     * @param closure_params Parameter list that contains species properties
     *
     */
    PlasmaSpeciesEOS(const panzer::IntegrationRule& ir,
                     const Teuchos::ParameterList& closure_params);

    /** \brief Evaluate the equation of state variables on the workset.
     *
     *  \param[in] workset  Evaluation data containing cell and point
     *                      information for the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /// String storing species
    std::string _species_name;

    // Species mass
    double _ms;

  public:
    /// Species pressure
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _p;

    /// Species momentum density
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _rho;

    /// Species energy density
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _E;

    /// Time derivative of species pressure
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _drhodt;

    /// Time derivative of species energy density
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dEdt;

  private:
    /// Species number density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nd;

    /// Species velocity
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, VelocityDim>
        _velocity;

    /// Species temperature
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _T;

    /// Time derivative of species number density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dnddt;

    /// Time derivative of species velocity
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, VelocityDim>
        _dveldt;

    /// Time derivative of species temperature
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dTdt;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_PLASMASPECIESEOS_HPP
