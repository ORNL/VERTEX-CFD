#ifndef VERTEXCFD_CLOSURE_PLASMASPECIESCONVECTIVEFLUX_HPP
#define VERTEXCFD_CLOSURE_PLASMASPECIESCONVECTIVEFLUX_HPP

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
 * @brief Compute the convective flux for the species equations (number density
 * equation, momentun density equation,
 * and energy density equation).
 *
 * @tparam EvalType Evaluation type (Residual, Jacobian, Tangent, Hessian).
 *
 * @tparam Traits Panzer traits for types (e.g. panzer::Traits).
 *
 * @tparam NumSpaceDim Dimension of the computational domain (2 or 3).

 * @see tstPlasmaSpeciesConvectiveFlux.cpp for examples.
*/
template<class EvalType, class Traits, int NumSpaceDim>
class PlasmaSpeciesConvectiveFlux
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
     * @param closure_params Parameter list that contains species name and
     * residual name.
     *
     * @param flux_prefix Flux prefix for boundary conditions.
     *
     * @param field_prefix Field prefix for boundary conditions.
     *
     */
    PlasmaSpeciesConvectiveFlux(const panzer::IntegrationRule& ir,
                                const Teuchos::ParameterList& closure_params,
                                const std::string& flux_prefix = "",
                                const std::string& field_prefix = "");

    /** \brief Evaluate the fluxes on the workset.
     *
     *  \param[in] workset  Evaluation data containing cell and point
     *                      information for the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /// String storing species name and residual name
    std::string _species_name;
    std::string _res_name;

  public:
    /// Flux for number density equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _nd_flux;

    /// Flux for momentum density equations
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _momentum_flux;

    /// Flux for energy density equation
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _energy_flux;

  private:
    /// Species number density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nd;
    /// Species velocity
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, VelocityDim>
        _velocity;
    /// Species density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    /// Species pressure
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _p;
    /// Species energy density
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _E;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_PLASMASPECIESCONVECTIVEFLUX_HPP
