#ifndef VERTEXCFD_CLOSURE_SOLIDELECTRICCONDUCTIVITY_HPP
#define VERTEXCFD_CLOSURE_SOLIDELECTRICCONDUCTIVITY_HPP

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
// Electric conductivity evaluation for solid domain.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class SolidElectricConductivity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    SolidElectricConductivity(const panzer::IntegrationRule& ir,
                              const Teuchos::ParameterList& closure_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sigma;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _electric_potential;

    double _sigma0;

    enum ElecCondType
    {
        constant,
        inverse_proportional
    };

    ElecCondType _elec_cond_type;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SOLIDELECTRICCONDUCTIVITY_HPP
