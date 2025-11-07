#ifndef VERTEXCFD_BOUNDARYSTATE_VARIABLEFIXED_HPP
#define VERTEXCFD_BOUNDARYSTATE_VARIABLEFIXED_HPP

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
class VariableFixed : public panzer::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    VariableFixed(const panzer::IntegrationRule& ir,
                  const Teuchos::ParameterList& bc_params,
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
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    double _fixed_value;
    int _num_grad_dim;
    double _a, _b;
    double _time_init, _time_final;
    double _variable_value;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VARIABLEFIXED_HPP
