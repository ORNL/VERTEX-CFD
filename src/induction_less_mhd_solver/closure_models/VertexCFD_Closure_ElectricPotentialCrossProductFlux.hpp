#ifndef VERTEXCFD_CLOSURE_ELECTRICPOTENTIALCROSSPRODUCTFLUX_HPP
#define VERTEXCFD_CLOSURE_ELECTRICPOTENTIALCROSSPRODUCTFLUX_HPP

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
// Multi-dimension cross product flux evaluation for electric potential
// equation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class ElectricPotentialCrossProductFlux
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int field_size = 3;

    ElectricPotentialCrossProductFlux(const panzer::IntegrationRule& ir,
                                      const std::string& flux_prefix = "",
                                      const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _electric_potential_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sigma;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_electric_potential;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  field_size>
        _ext_magn_field;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_ELECTRICPOTENTIALCROSSPRODUCTFLUX_HPP
