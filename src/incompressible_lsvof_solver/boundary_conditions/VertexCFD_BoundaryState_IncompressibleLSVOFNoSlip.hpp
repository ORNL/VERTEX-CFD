#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFNOSLIP_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFNOSLIP_HPP

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFNoSlip
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFNoSlip(const panzer::IntegrationRule& ir,
                              const int& num_lsvof_dofs,
                              const std::string& lsvof_model_name,
                              const std::string& continuity_model_name,
                              const bool& build_mom_equ);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    Teuchos::RCP<PHX::DataLayout> _phase_layout;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _boundary_lagrange_pressure;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_lagrange_pressure;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _boundary_velocity;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _boundary_grad_velocity;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _boundary_alphas;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_phi;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _alphas;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _phi;

    enum LSVOFModelType
    {
        VOF,
        CLS
    };

    LSVOFModelType _lsvof_model_type;
    bool _is_edac;
    bool _build_mom_equ;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFNOSLIP_HPP
