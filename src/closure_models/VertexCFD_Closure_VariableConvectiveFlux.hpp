#ifndef VERTEXCFD_CLOSURE_VARIABLECONVECTIVEFLUX_HPP
#define VERTEXCFD_CLOSURE_VARIABLECONVECTIVEFLUX_HPP

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
// Conservative term for convective-diffusion equation
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class VariableConvectiveFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    VariableConvectiveFlux(const panzer::IntegrationRule& ir,
                           const Teuchos::ParameterList& closure_params,
                           const std::string& flux_prefix = "",
                           const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    std::string _variable_name;
    std::string _equation_name;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _var_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _var;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VARIABLECONVECTIVEFLUX_HPP
