#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_HPP

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

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
// Multi-dimension viscous flux evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleViscousFlux
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleViscousFlux(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params,
        const bool use_turbulence_model,
        const std::string& flux_prefix = "",
        const std::string& gradient_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _continuity_flux;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _energy_flux;
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _momentum_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _cp;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _k;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_press;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temp;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    double _gamma;
    double _beta;
    bool _solve_temp;
    bool _use_turbulence_model;
    std::string _continuity_model_name;
    bool _is_edac;
    double _Pr_t;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEVISCOUSFLUX_HPP
