#ifndef VERTEXCFD_CLOSURE_JOULEHEATINGSOURCE_HPP
#define VERTEXCFD_CLOSURE_JOULEHEATINGSOURCE_HPP

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
// Joule heating source term
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class JouleHeatingSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    JouleHeatingSource(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _jh_source;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sigma;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _electric_current_density;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_JOULEHEATINGSOURCE_HPP
