#ifndef VERTEXCFD_BOUNDARYSTATE_VARIABLEEXTRAPOLATE_HPP
#define VERTEXCFD_BOUNDARYSTATE_VARIABLEEXTRAPOLATE_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class VariableExtrapolate : public panzer::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    VariableExtrapolate(const panzer::IntegrationRule& ir,
                        const std::string variable_name);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_variable;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_variable;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _variable;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    int _num_grad_dim;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VARIABLEEXTRAPOLATE_HPP
