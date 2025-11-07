#ifndef VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Volumetric source/sink for conduction.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConductionVolumetricSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ConductionVolumetricSource(const panzer::IntegrationRule& ir,
                               const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _source;

  private:
    int _num_space_dim;
    int _ir_degree;
    double _q;
    double _x_left;
    double _x_right;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

    enum HeatSourceType
    {
        constant,
        xlinear
    };

    HeatSourceType _heat_source_type;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_HPP
