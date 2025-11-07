#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_HPP

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

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
// Diffusivity coefficients for K-Tau turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleKTauDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleKTauDiffusivityCoefficient(
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
    double _sigma_t;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_t;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_HPP
