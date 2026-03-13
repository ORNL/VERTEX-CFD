#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLETIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLETIMEDERIVATIVE_HPP

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
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleTimeDerivative
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleTimeDerivative(const panzer::IntegrationRule& ir,
                                 const Teuchos::ParameterList& fluid_params);

    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_continuity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_energy;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _dqdt_momentum;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _dxdt_lagrange_pressure;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _cp;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _dxdt_velocity;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_temperature;

    bool _solve_temp;
    double _beta;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLETIMEDERIVATIVE_HPP
