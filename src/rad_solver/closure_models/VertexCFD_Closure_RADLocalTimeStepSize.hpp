#ifndef VERTEXCFD_CLOSURE_RADLOCALTIMESTEPSIZE_HPP
#define VERTEXCFD_CLOSURE_RADLOCALTIMESTEPSIZE_HPP

#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

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
// Compute local time step size based on the mesh size and maximum eigenvalue.
// The time step size used by the solver and based on the CFL condition is
// computed in the observer
// 'VertexCFD_TempusTimeStepControl_GlobalCFL_impl.hpp'
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class RADLocalTimeStepSize : public panzer::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    RADLocalTimeStepSize(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop);

    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_species;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _local_dt;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;
    Kokkos::View<double**, Kokkos::LayoutLeft, PHX::mem_space> _bateman_matrix;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    bool _build_reaction;
    bool _build_advection;
    bool _build_diffusion;
    double _nu;

    enum TmpVars
    {
        LOCAL_DT_REACT,
        LOCAL_DT_ADV,
        LOCAL_DT_DIF,
        NUM_TMPS
    };

    /// View type for shared memory
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADLOCALTIMESTEPSIZE_HPP
