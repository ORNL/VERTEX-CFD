#ifndef VERTEXCFD_CLOSURE_CONDUCTIONFLUX_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONFLUX_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Thermal flux evaluation for conduction.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConductionFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ConductionFlux(const panzer::IntegrationRule& ir,
                   const std::string& flux_prefix = "",
                   const std::string& gradient_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _conduction_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _thermal_conductivity;

    int _num_space_dim;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONFLUX_HPP
