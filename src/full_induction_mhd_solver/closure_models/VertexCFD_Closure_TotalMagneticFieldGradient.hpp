#ifndef VERTEXCFD_CLOSURE_TOTALMAGNETICFIELDGRADIENT_HPP
#define VERTEXCFD_CLOSURE_TOTALMAGNETICFIELDGRADIENT_HPP

#include "utils/VertexCFD_Utils_MagneticDim.hpp"

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
// Compute total magnetic field gradient for full induction MHD
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class TotalMagneticFieldGradient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    TotalMagneticFieldGradient(const panzer::IntegrationRule& ir,
                               const std::string& gradient_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    bool _uniform_external_field;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_induced_magnetic_field;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_magnetic_field_dim>
        _grad_external_magnetic_field;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim, MagneticDim>
        _grad_total_magnetic_field;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _divergence_total_magnetic_field;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_TOTALMAGNETICFIELDGRADIENT_HPP
