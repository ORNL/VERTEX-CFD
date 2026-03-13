#ifndef VERTEXCFD_CLOSURE_VARIABLEOLDVALUE_IMPL_HPP
#define VERTEXCFD_CLOSURE_VARIABLEOLDVALUE_IMPL_HPP

#include "utils/VertexCFD_Utils_GetValue.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
VariableOldValue<EvalType, Traits>::VariableOldValue(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _old_value_type(OldValueType::LastTime)
    , _make_grad(true)
    , _field_name(closure_params.get<std::string>("Field Name"))
    , _prefix("OLD_")
    , _nl_iter(-1)
    , _stage(0)
    , _time(0.0)
    , _update(false)
    , _workset_id(0)
    , _var(_field_name, ir.dl_scalar)
    , _grad_var("GRAD_" + _field_name, ir.dl_vector)
{
    // Check old value type
    if (closure_params.isType<std::string>("Old Value Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<OldValueType>(
                Teuchos::tuple<std::string>("LastTime", "LastStage"),
                "LastTime"));

        _old_value_type = type_validator->getIntegralValue(
            closure_params.get<std::string>("Old Value Type"));
    }

    // Set suffix
    if (_old_value_type == OldValueType::LastStage)
        _prefix = "STAR_";

    // Check for gradient
    if (closure_params.isType<bool>("Save Old Gradient"))
        _make_grad = closure_params.get<bool>("Save Old Gradient");

    // Initialize old fields
    _var_old_val = PHX::MDField<double, panzer::Cell, panzer::Point>(
        _prefix + _field_name, ir.dl_scalar);
    if (_make_grad)
        _grad_var_old_val
            = PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim>(
                _prefix + "GRAD_" + _field_name, ir.dl_vector);

    // Add evaludated fields
    this->addEvaluatedField(_var_old_val);
    if (_make_grad)
        this->addEvaluatedField(_grad_var_old_val);

    // Add dependent fields
    this->addDependentField(_var);
    if (_make_grad)
        this->addDependentField(_grad_var);

    // Closure model name
    this->setName(_field_name + " Old Value");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void VariableOldValue<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _num_worksets = (*sd.worksets_).size();

    // Initialize the size of the vector views
    _var_old_val_vec
        = Kokkos::View<double***, PHX::mem_space>("var_old_val_vec",
                                                  _num_worksets,
                                                  _var_old_val.extent(0),
                                                  _var_old_val.extent(1));

    if (_make_grad)
    {
        _grad_var_old_val_vec = Kokkos::View<double****, PHX::mem_space>(
            "grad_var_old_val_vec",
            _num_worksets,
            _grad_var_old_val.extent(0),
            _grad_var_old_val.extent(1),
            _grad_var_old_val.extent(2));
    }

    // size the iterator vectors for storing intial workset cell data
    _workset_id.resize(_num_worksets);

    // Loop over worksets to fill workset id vector
    for (std::size_t wks = 0; wks < (*sd.worksets_).size(); ++wks)
    {
        _current_workset = wks;
        const panzer::Workset workset = (*sd.worksets_)[wks];
        // If the workset is empty, skip the iterator setup
        if (workset.num_cells > 0)
        {
            _workset_id[wks] = workset.getIdentifier();
        }
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void VariableOldValue<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Check for update on 1st workset only
    if (workset.getIdentifier() == _workset_id[0])
    {
        ++_nl_iter;

        // Default: no update
        _update = false;

        const double curr_time = workset.time;
        const double curr_stage = workset.stage_number;

        // If new time or new stage, reset non-linear iteration counter and
        // update
        if ((curr_stage > _stage) || (curr_time > _time))
        {
            _nl_iter = 0;
            _stage = curr_stage;
            _time = curr_time;
        }

        // Update last time values on 0th NL iteration of 0th stage
        if ((_old_value_type == OldValueType::LastTime) && (_stage == 0)
            && (_nl_iter == 0))
        {
            _update = true;
        }
        // Update last stage values on 0th NL iteration
        else if ((_old_value_type == OldValueType::LastStage)
                 && (_nl_iter == 0))
        {
            _update = true;
        }

        // Don't update on Jacobian eval
        if constexpr (Sacado::IsADType<scalar_type>::value)
        {
            _update = false;
        }
    }

    // Get current workset index
    for (int i = 0; i < _num_worksets; ++i)
    {
        if (_workset_id[i] == workset.getIdentifier())
        {
            _current_workset = i;
            break;
        }
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void VariableOldValue<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _var.extent(1);
    const int num_space_dim = _make_grad ? _grad_var.extent(2) : 0;

    // Set field values to stored values
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Update old values when criteria are met
            if (_update)
            {
                using Utils::getValue;

                _var_old_val_vec(_current_workset, cell, point)
                    = getValue(_var(cell, point));

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    _grad_var_old_val_vec(_current_workset, cell, point, dim)
                        = getValue(_grad_var(cell, point, dim));
                }
            }

            _var_old_val(cell, point)
                = _var_old_val_vec(_current_workset, cell, point);

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _grad_var_old_val(cell, point, dim) = _grad_var_old_val_vec(
                    _current_workset, cell, point, dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VARIABLEOLDVALUE_IMPL_HPP
