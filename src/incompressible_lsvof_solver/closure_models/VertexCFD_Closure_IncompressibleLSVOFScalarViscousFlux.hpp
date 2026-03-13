#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARVISCOUSFLUX_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARVISCOUSFLUX_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension evaluation for LSVOF viscous flux
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFScalarViscousFlux
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFScalarViscousFlux(const panzer::IntegrationRule& ir,
                                         const std::string& scalar_name,
                                         const std::string& equation_name,
                                         const std::string& flux_prefix = "",
                                         const std::string& gradient_prefix
                                         = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _scalar_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _artificial_viscosity;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_scalar;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARVISCOUSFLUX_HPP
