#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_HPP

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
// Diffusion coefficients for standard K-Epsilon turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleKEpsilonDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleKEpsilonDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const double sigma_k = 1.0,
        const double sigma_e = 1.3,
        const std::string field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    double _sigma_k;
    double _sigma_e;
    int _num_grad_dim;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_e;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_HPP
