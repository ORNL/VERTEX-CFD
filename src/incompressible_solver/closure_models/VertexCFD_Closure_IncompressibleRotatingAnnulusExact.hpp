#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEROTATINGANNULUSEXACT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEROTATINGANNULUSEXACT_HPP

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
// Exact solution for rotating annulus with viscous heating
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleRotatingAnnulusExact
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleRotatingAnnulusExact(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<double, panzer::Cell, panzer::Point> _temperature;
    PHX::MDField<double, panzer::Cell, panzer::Point> _lagrange_pressure;
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        _velocity;

  private:
    double _nu;
    double _rho;
    double _k;
    double _ro;
    double _ri;
    double _kappa;
    double _omega;
    double _To;
    double _Ti;
    double _N;

    int _ir_degree;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEROTATINGANNULUSEXACT_HPP
