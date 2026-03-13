#ifndef VERTEXCFD_CLOSURE_VARIABLEOLDVALUE_HPP
#define VERTEXCFD_CLOSURE_VARIABLEOLDVALUE_HPP

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
// Generic class to store old values of PHX::MDFields
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class VariableOldValue : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    VariableOldValue(const panzer::IntegrationRule& ir,
                     const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    enum OldValueType
    {
        LastTime,
        LastStage
    };

    OldValueType _old_value_type;
    bool _make_grad;
    std::string _field_name;
    std::string _prefix;
    int _nl_iter;
    int _stage;
    double _time;
    bool _update;
    int _num_worksets;
    std::vector<size_t> _workset_id;
    std::size_t _current_workset = 0;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _var;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_var;

    Kokkos::View<double***, PHX::mem_space> _var_old_val_vec;
    Kokkos::View<double****, PHX::mem_space> _grad_var_old_val_vec;

  public:
    PHX::MDField<double, panzer::Cell, panzer::Point> _var_old_val;
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_var_old_val;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VARIABLEOLDVALUE_HPP
