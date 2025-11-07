#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSURFACETENSIONFORCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSURFACETENSIONFORCE_HPP

#include "utils/VertexCFD_Utils_Constants.hpp"

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
// Multi-dimension evaluation for LSVOF surface tension force
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFSurfaceTensionForce
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFSurfaceTensionForce(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::vector<std::string>& phase_names,
        const std::string& flux_prefix = "",
        const std::string& gradient_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _surface_tension_flux;

  private:
    double _sigma_alpha;
    int num_phases;

    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        VertexCFD::Constants::MAX_NUM_VIEW>
        _phase_grad;

    /// View type for shared memory
    using scratch_view
        = Kokkos::View<scalar_type*,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

    using scratch_view_Fs
        = Kokkos::View<scalar_type*,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

    using scratch_view_normal
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSURFACETENSIONFORCE_HPP
