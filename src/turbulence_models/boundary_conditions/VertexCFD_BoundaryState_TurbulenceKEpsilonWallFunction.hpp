#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_HPP

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
// Wall function boundary conditions for K-Epsilon family of turbulence
// models as outlined by Kuzmin et al. (2007)
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class TurbulenceKEpsilonWallFunction
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    TurbulenceKEpsilonWallFunction(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& bc_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_e;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_k;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_e;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_u_tau;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_y_plus;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _wall_func_nu_t;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _k;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _e;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_k;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_e;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    int _num_grad_dim;
    double _C_mu;
    double _kappa;
    double _yp_tr;
    bool _neumann;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_HPP
