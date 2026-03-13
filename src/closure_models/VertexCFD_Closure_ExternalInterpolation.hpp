#ifndef VERTEXCFD_CLOSURE_EXTERNALINTERPOLATION_HPP
#define VERTEXCFD_CLOSURE_EXTERNALINTERPOLATION_HPP

#ifdef VERTEXCFD_HAVE_ARBORX

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <ArborX.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Closure model to interpolate a named scalar field from an external Exodus
// file
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class ExternalInterpolation : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    using ExecutionSpace = PHX::exec_space;
    using MemorySpace = ExecutionSpace::memory_space;

    ExternalInterpolation(const panzer::IntegrationRule& ir,
                          const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _scalar_field;

  private:
    void read_node_values(size_t time_step);

    int _ir_degree;
    int _ir_index;

    size_t _last_time_step_index = 0;
    int _var_index;

    std::string _file_name;
    std::string _name;
    std::vector<double> _time_values;

    std::vector<size_t> _reduced_to_global;

#if ARBORX_VERSION_MAJOR < 2
    using ArborXPoint
        = ArborX::ExperimentalHyperGeometry::Point<NumSpaceDim, double>;
#else
    using ArborXPoint = ArborX::Point<NumSpaceDim, double>;
#endif
    Kokkos::View<ArborXPoint*, MemorySpace> _source_points;
    Kokkos::View<double*, MemorySpace> _nodal_values;
    Kokkos::View<double*, Kokkos::HostSpace> _nodal_values_host;
    Kokkos::View<double*, Kokkos::HostSpace> _nodal_values_host_reduced;
    Kokkos::View<double*, MemorySpace> _interpolated_values;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_HAVE_ARBORX

#endif // end VERTEXCFD_CLOSURE_EXTERNALINTERPOLATION_HPP
