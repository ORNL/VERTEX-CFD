#ifndef VERTEXCFD_CLOSURE_RADEXACTSOLUTION_HPP
#define VERTEXCFD_CLOSURE_RADEXACTSOLUTION_HPP

#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"
#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Reaction Advection Diffusion equation exact solution.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class RADExactSolution : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    RADExactSolution(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop,
        const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_species;

  public:
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _exact_species;

  private:
    int _ir_degree;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    double _nu;
    Kokkos::View<double**, Kokkos::LayoutLeft, PHX::mem_space> _bateman_matrix;
    Kokkos::Array<double, VertexCFD::Constants::MAX_NUM_VIEW> _scale;
    bool _build_reaction;
    bool _build_advection;
    bool _build_diffusion;
    double _time;
    double _sigma;
    double _init_loc;

    enum TmpVars
    {
        GAUSSIAN,
        DENUM,
        SUM_K,
        PROD_L,
        LMBD_PROD,
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

} // namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_RADEXACTSOLUTION_HPP
