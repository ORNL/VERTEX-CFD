#ifndef VERTEXCFD_CLOSURE_RADFISSIONSOURCE_HPP
#define VERTEXCFD_CLOSURE_RADFISSIONSOURCE_HPP

#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"
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
// Fission source term for reaction advection diffusion equations
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class RADFissionSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    RADFissionSource(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop,
        const std::string& flux_name);

    void evaluateFields(typename Traits::EvalData d) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_species;

  public:
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _fission_source;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _flux;
    Kokkos::View<double*, Kokkos::LayoutLeft, PHX::mem_space> _gamma;
    double _xs;
    double _avagadro;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADFISSIONSOURCE_HPP
