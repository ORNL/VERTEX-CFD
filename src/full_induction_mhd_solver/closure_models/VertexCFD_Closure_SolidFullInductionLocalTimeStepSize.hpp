#ifndef VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONLOCALTIMESTEPSIZE_HPP
#define VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONLOCALTIMESTEPSIZE_HPP

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include "utils/VertexCFD_Utils_MagneticDim.hpp"

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
// Compute local time step size based on the mesh size and maximum eigenvalue
// for the solid induction MHD equation set.
// The time step size used by the solver and based on the CFL condition is
// computed in the observer
// 'VertexCFD_TempusTimeStepControl_GlobalCFL_impl.hpp'
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class SolidFullInductionLocalTimeStepSize
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    SolidFullInductionLocalTimeStepSize(
        const panzer::IntegrationRule& ir,
        const MHDProperties::FullInductionMHDProperties& mhd_props);

    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _local_dt;

  private:
    double _magnetic_permeability;
    double _c_h;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;
    PHX::MDField<const double, panzer::Cell, panzer::Point> _solid_density;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, MagneticDim>
        _total_magnetic_field;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONLOCALTIMESTEPSIZE_HPP
