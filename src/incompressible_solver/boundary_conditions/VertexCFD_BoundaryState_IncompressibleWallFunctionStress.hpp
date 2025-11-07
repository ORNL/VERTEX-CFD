#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEWALLFUNCTIONSTRESS_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEWALLFUNCTIONSTRESS_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleWallFunctionStress
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleWallFunctionStress(const panzer::IntegrationRule& ir,
                                     const std::string& flux_prefix);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _wall_func_shear_stress;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _boundary_u_tau;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _boundary_y_plus;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEWALLFUNCTIONSTRESS_HPP
