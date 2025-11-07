#ifndef VERTEXCFD_CLOSURE_TOTALMAGNETICFIELD_HPP
#define VERTEXCFD_CLOSURE_TOTALMAGNETICFIELD_HPP

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
// Compute total magnetic field for full induction MHD
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class TotalMagneticField : public panzer::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    TotalMagneticField(const panzer::IntegrationRule& ir,
                       const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    bool _uniform_external_field;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _induced_magnetic_field;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_magnetic_field_dim>
        _external_magnetic_field;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim>
        _total_magnetic_field;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_TOTALMAGNETICFIELD_HPP
