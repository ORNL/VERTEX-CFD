#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVARIABLEPROPERTIES_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVARIABLEPROPERTIES_HPP

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Field.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension variable property evaluation for LSVOF Navier-Stokes
// equations
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleLSVOFVariableProperties
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    using view_layout = typename PHX::DevLayout<scalar_type>::type;

    IncompressibleLSVOFVariableProperties(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& lsvof_params,
        const std::vector<std::string>& phase_names,
        const bool& build_dxdts = true,
        const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _mu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dxdt_rho;

  private:
    enum LSVOFModelType
    {
        VOF,
        CLS
    };

    LSVOFModelType _lsvof_model_type;
    bool _build_dxdts;
    Teuchos::RCP<PHX::DataLayout> _phase_layout;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _alphas;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _dxdt_alphas;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _alpha_n;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _heaviside;
    Kokkos::View<double*, view_layout, PHX::mem_space> _phase_rho;
    Kokkos::View<double*, view_layout, PHX::mem_space> _phase_mu;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVARIABLEPROPERTIES_HPP
