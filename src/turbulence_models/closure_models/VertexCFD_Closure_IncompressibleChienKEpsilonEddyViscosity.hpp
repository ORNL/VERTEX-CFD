#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
// Turbulent eddy viscosity for Chien's low-Re K-Epsilon turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleChienKEpsilonEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleChienKEpsilonEddyViscosity(
        const panzer::IntegrationRule& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& user_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;

    Teuchos::RCP<panzer::GlobalData> _global_data;

    double _C_nu;
    double _C_tau;
    int _num_grad_dim;
    double _area;

    scalar_type _wall_shear_stress;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_HPP
