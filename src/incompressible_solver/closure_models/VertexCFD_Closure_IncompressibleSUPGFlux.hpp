#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGFLUX_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGFLUX_HPP

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
// Multi-dimension SUPG flux evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleSUPGFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleSUPGFlux(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& closure_params);

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
    bool _use_source;
    std::string _source_prefix;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _momentum_source;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _cp;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _tau_supg_cont;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _tau_supg_mom;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _tau_supg_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_temperature;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _energy_source;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _dxdt_velocity;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    bool _solve_temp;

    enum TmpVars
    {
        VEL_RES,
        TEMP_RES,
        NUM_TMPS
    };

    /// View type for shared memory
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGFLUX_HPP
