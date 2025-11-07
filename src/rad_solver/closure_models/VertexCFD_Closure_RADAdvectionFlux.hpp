#ifndef VERTEXCFD_CLOSURE_RADADVECTIONFLUX_HPP
#define VERTEXCFD_CLOSURE_RADADVECTIONFLUX_HPP

#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"
#include "utils/VertexCFD_Utils_Constants.hpp"

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
// Advection term for for reaction advection diffusion equations
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class RADAdvectionFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    RADAdvectionFlux(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop,
        const std::string& flux_prefix = "",
        const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_species;

  public:
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        VertexCFD::Constants::MAX_NUM_VIEW>
        _species_flux;

  private:
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _species;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADADVECTIONFLUX_HPP
