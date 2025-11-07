#ifndef VERTEXCFD_CLOSURE_MAGNETICPRESSURE_HPP
#define VERTEXCFD_CLOSURE_MAGNETICPRESSURE_HPP

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

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
// Magnetic pressure for full induction MHD
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class MagneticPressure : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    MagneticPressure(const panzer::IntegrationRule& ir,
                     const MHDProperties::FullInductionMHDProperties& mhd_props);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    double _magnetic_permeability;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, MagneticDim>
        _total_magnetic_field;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _magnetic_pressure;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_MAGNETICPRESSURE_HPP
