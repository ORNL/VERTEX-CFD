#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFNTHPHASEFRACTION_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFNTHPHASEFRACTION_HPP

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Field.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Evaluator for the Nth phase fraction of a multiphase system, for which
// there is no transport equation
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleLSVOFNthPhaseFraction
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleLSVOFNthPhaseFraction(
        const panzer::IntegrationRule& ir,
        const std::vector<std::string>& phase_names,
        const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _alpha_n;

  private:
    Teuchos::RCP<PHX::DataLayout> _phase_layout;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _alphas;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFNTHPHASEFRACTION_HPP
