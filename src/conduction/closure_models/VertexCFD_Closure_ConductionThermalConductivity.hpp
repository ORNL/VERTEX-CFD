#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_HPP

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
// Temperature-dependent thermal conductivity evaluation for conduction.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConductionThermalConductivity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ConductionThermalConductivity(const panzer::IntegrationRule& ir,
                                  const Teuchos::ParameterList& closure_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _thermal_conductivity;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;

    int _num_space_dim;
    double _k0;

    enum ThermCondType
    {
        constant,
        one_over_temp
    };

    ThermCondType _therm_cond_type;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_HPP
