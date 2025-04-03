#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_HPP

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
// Diffusivity coefficients for Menter's SST K-Omega turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleSSTDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleSSTDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _sst_blending_function;

    double _nu;
    double _sigma_k1;
    double _sigma_k2;
    double _sigma_w1;
    double _sigma_w2;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_w;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP
