#ifndef VERTEXCFD_CLOSURE_RADFISSIONSOURCEEXACTSOLUTION_HPP
#define VERTEXCFD_CLOSURE_RADFISSIONSOURCEEXACTSOLUTION_HPP

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
// Fission source term exact solution.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class RADFissionSourceExactSolution
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    RADFissionSourceExactSolution(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop,
        const Teuchos::ParameterList& closure_params);

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
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _neutron_flux;

  private:
    double _xs;
    double _avagadro;
    Kokkos::View<double*, Kokkos::LayoutLeft, PHX::mem_space> _gamma;
    double _a;
    double _b;
    double _kappa;
    double _beta;
    double _flux_amp;
    bool _calc_flux;
    Kokkos::Array<double, VertexCFD::Constants::MAX_NUM_VIEW> _initial_amp;
    double _time;

    enum TmpVars
    {
        CONSTS,
        CONSTC,
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

#endif // VERTEXCFD_CLOSURE_RADFISSIONSOURCEEXACTSOLUTION_HPP
