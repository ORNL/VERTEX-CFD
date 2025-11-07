#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECONSTANTSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECONSTANTSOURCE_HPP

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
// Multi-dimension constant source evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleConstantSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleConstantSource(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& closure_params);

    void evaluateFields(typename Traits::EvalData /* workset */) override;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _momentum_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _energy_source;
    Teuchos::RCP<panzer::GlobalData> _global_data;

  private:
    bool _solve_temp;
    bool _solve_ind_less_mhd;
    Kokkos::Array<scalar_type, num_space_dim> _mom_input_source;
    double _energy_input_source;

    enum FlowDirection
    {
        x,
        y,
        z
    };

    FlowDirection _flow_direction;
    int _flow_direction_idx;

    bool _const_vol_flow;
    scalar_type _m_target;
    scalar_type _bottom_wall_area;
    scalar_type _inlet_wall_area;
    scalar_type _m_bc;
    scalar_type _wall_shear_stress;
    scalar_type _lorentz_force;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECONSTANTSOURCE_HPP
