#ifndef VERTEXCFD_CLOSURE_RADERRORNORMS_HPP
#define VERTEXCFD_CLOSURE_RADERRORNORMS_HPP

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
// Compute error norms between exact solution and numerical solutions
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class RADErrorNorms : public panzer::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    RADErrorNorms(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    int _num_species;

  public:
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _L1_error_species;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _L2_error_species;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _volume;

  private:
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _exact_species;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  VertexCFD::Constants::MAX_NUM_VIEW>
        _species;
};

//---------------------------------------------------------------------------//

} // namespace ClosureModel
} // end namespace VertexCFD

#include "VertexCFD_Closure_RADErrorNorms_impl.hpp"

#endif // VERTEXCFD_CLOSURE_RADERRORNORMS_HPP
