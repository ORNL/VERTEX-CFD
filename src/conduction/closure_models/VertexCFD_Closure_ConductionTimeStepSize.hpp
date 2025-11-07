#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Compute time-step size for conduction equation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConductionTimeStepSize : public panzer::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ConductionTimeStepSize(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _local_dt;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _thermal_conductivity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _specific_heat;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _solid_density;
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_HPP
