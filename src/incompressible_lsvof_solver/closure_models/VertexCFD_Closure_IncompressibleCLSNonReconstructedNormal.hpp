#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSNONRECONSTRUCTEDNORMAL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSNONRECONSTRUCTEDNORMAL_HPP

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
// Non-reconstructed interface normal for conservative level set model testing.
// See Quezada de Luna (2019)
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleCLSNonReconstructedNormal
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleCLSNonReconstructedNormal(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _q;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_phi_0;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSNONRECONSTRUCTEDNORMAL_HPP
