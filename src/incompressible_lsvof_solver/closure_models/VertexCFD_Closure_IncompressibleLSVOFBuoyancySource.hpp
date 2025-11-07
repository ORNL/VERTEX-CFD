#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFBUOYANCYSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFBUOYANCYSOURCE_HPP

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
// Buoyancy source evaluation for LSVOF.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFBuoyancySource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFBuoyancySource(const panzer::IntegrationRule& ir,
                                      const Teuchos::ParameterList& user_params,
                                      const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _buoyancy_momentum_source;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    Kokkos::Array<double, num_space_dim> _gravity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFBUOYANCYSOURCE_HPP
