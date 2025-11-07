#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFTIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFTIMEDERIVATIVE_HPP

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
// Time derivative evaluation for LSVOF Navier-Stokes equations
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFTimeDerivative
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFTimeDerivative(const panzer::IntegrationRule& ir,
                                      const Teuchos::ParameterList& lsvof_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_continuity;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _dqdt_momentum;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_pressure;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_density;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _dxdt_velocity;

    double _beta;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFTIMEDERIVATIVE_HPP
