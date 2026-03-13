#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSTIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSTIMEDERIVATIVE_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
// Time for conservative level set method. See Quezada de Luna (2019). The time
// component of the residual is not equal to d(phi)/dt, so here we calculate it
// manually from the sign function.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleCLSTimeDerivative
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleCLSTimeDerivative(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_sign;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sign;
    PHX::MDField<const double, panzer::Cell, panzer::Point> _star_sign;

    double _dt;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSTIMEDERIVATIVE_HPP
