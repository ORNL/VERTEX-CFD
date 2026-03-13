#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEBUOYANCYSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEBUOYANCYSOURCE_HPP

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
// Thermal buoyancy source evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleBuoyancySource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleBuoyancySource(const panzer::IntegrationRule& ir,
                                 const Teuchos::ParameterList& closure_params,
                                 const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _buoyancy_continuity_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _buoyancy_energy_source;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _buoyancy_momentum_source;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;
    double _beta_T;
    double _T_ref;
    Kokkos::Array<double, num_space_dim> _gravity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEBUOYANCYSOURCE_HPP
