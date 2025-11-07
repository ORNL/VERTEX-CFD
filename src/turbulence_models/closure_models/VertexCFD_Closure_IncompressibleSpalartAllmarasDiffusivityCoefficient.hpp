#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_HPP

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
// Diffusion coefficient for Spalart-Allmaras turbulence model (SA-neg)
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleSpalartAllmarasDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleSpalartAllmarasDiffusivityCoefficient(
        const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sa_var;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    const double _cn1;
    const double _sigma;
    const scalar_type _one;
    const int _num_grad_dim;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_HPP
