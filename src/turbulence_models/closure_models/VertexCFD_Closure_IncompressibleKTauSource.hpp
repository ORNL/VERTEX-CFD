#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_HPP

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
// Source term for the K-Tau turbulence model with/without
// limiter
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKTauSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleKTauSource(const panzer::IntegrationRule& ir,
                             const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    double _beta_star;
    double _gamma;
    double _beta_0;
    double _sigma_d;
    double _sigma_t;
    bool _limit_production;
    bool _limit_destruction;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_prod;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_dest_nut;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_dest_nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_cross;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_HPP
