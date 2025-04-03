#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class TurbulenceKOmegaWallResolved
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    TurbulenceKOmegaWallResolved(const panzer::IntegrationRule& ir,
                                 const Teuchos::ParameterList& bc_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_w;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_w;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _k;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_k;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_w;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;

    double _time;
    double _omega_wall;
    double _omega_wall_init;
    double _omega_ramp_time;
    int _num_grad_dim;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_HPP
