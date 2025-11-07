#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP

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
// Diffusivity coefficients for the Wilcox (2006) K-Omega turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleKOmegaDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleKOmegaDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    double _sigma_k;
    double _sigma_w;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_w;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP
