#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_HPP

#include "VertexCFD_Utils_PhaseIndex.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_DataLayout.hpp>
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
// Multi-dimension convective flux evaluation for LSVOF scalar equation
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFScalarConvectiveFlux
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFScalarConvectiveFlux(const panzer::IntegrationRule& ir,
                                            const int& phase_index,
                                            const int& num_lsvof_dofs,
                                            const std::string& equation_name,
                                            const std::string& flux_prefix = "",
                                            const std::string& field_prefix
                                            = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _scalar_flux;

  private:
    int _phase_index;
    Teuchos::RCP<PHX::DataLayout> _phase_layout;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, PhaseIndex>
        _volume_fractions;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARCONVECTIVEFLUX_HPP
