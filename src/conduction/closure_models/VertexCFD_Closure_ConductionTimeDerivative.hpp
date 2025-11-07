#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConductionTimeDerivative : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ConductionTimeDerivative(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_energy;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _density;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _specific_heat;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_temperature;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_HPP
