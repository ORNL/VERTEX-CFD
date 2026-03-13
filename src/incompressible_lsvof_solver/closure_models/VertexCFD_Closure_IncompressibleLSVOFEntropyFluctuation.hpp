#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFLUCTUATION_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFLUCTUATION_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
// Multi-dimension evaluation for LSVOF entropy fluctuation term
// evaluated as E(phi) - \overbar(E(phi))
//
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFEntropyFluctuation
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFEntropyFluctuation(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& equation_name,
        const Teuchos::RCP<panzer::GlobalData>& global_data);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _entropy_fluctuation;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _entropy_function;

    double _domain_volume;
    Teuchos::RCP<panzer::GlobalData> _global_data;
    scalar_type _entropy_int;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFLUCTUATION_HPP
