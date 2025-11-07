#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_HPP

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
// Source term for Spalart-Allmaras turbulence model (SA-neg)
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleSpalartAllmarasSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleSpalartAllmarasSource(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sa_var;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_sa_var;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    const double _sigma;
    const double _kappa;
    const double _c_b1;
    const double _c_b2;
    const double _c_t3;
    const double _c_t4;
    const double _c_v1;
    const double _c_v2;
    const double _c_v3;
    const double _c_w1;
    const double _c_w2;
    const double _c_w3;
    const scalar_type _rlim;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sa_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sa_prod;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sa_dest;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_HPP
