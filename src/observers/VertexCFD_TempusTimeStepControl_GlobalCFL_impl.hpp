#ifndef VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALCFL_IMPL_HPP
#define VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALCFL_IMPL_HPP

#include "VertexCFD_TempusTimeStepControl_GlobalCFL.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <algorithm>
#include <string>

namespace VertexCFD
{
namespace TempusTimeStepControl
{
//---------------------------------------------------------------------------//
template<class Scalar>
GlobalCFL<Scalar>::GlobalCFL(const Teuchos::ParameterList& user_params,
                             Teuchos::RCP<PhysicsManager> physics_manager)
    : _cfl(user_params.get<double>("CFL"))
    , _cfl_init(user_params.isType<double>("CFL_init")
                    ? user_params.get<double>("CFL_init")
                    : _cfl)
    , _cfl_transition_init(0.0)
    , _cfl_transition(0.0)
    , _cfl_type(CflTransitionType::steps)
    , _response_manager(physics_manager)
{
    // Validate transition type if present
    if (user_params.isType<std::string>("CFL_transition_type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<CflTransitionType>(
                Teuchos::tuple<std::string>("steps", "time"), "steps"));
        _cfl_type = type_validator->getIntegralValue(
            user_params.get<std::string>("CFL_transition_type"));
    }

    // Check for ramping options
    if (user_params.isType<double>("CFL_transition"))
    {
        _cfl_transition = user_params.get<double>("CFL_transition");
        if (user_params.isType<double>("CFL_transition_init"))
        {
            _cfl_transition_init
                = user_params.get<double>("CFL_transition_init");
        }
    }
    else if (user_params.isType<double>("CFL_transition_init"))
    {
        const std::string msg
            = "\n\n'CFL_transition_init' must be"
              "specified with 'CFL_transition'\n"
              "in the input file. Please add 'CFL_transition'\n"
              "to your input file.\n";
        throw std::runtime_error(msg);
    }

    this->setCurrentCFL(_cfl_init);

    _response_manager.addMinValueResponse("global_cfl_time_step", "local_dt");

    // For the new Tempus::TimeStepControlStrategy interface, we need to
    // set a few base class member variables. In particular, incorrect
    // behavior may result if "stepType_" is not set to "Variable".
    this->stepType_ = "Variable";
    this->strategyType_ = "CFL Strategy";
    this->name_ = "CFL Strategy";
}

//---------------------------------------------------------------------------//
// Determine the time step size.
template<class Scalar>
void GlobalCFL<Scalar>::setNextTimeStep(
    const Tempus::TimeStepControl<Scalar>& tsc,
    Teuchos::RCP<Tempus::SolutionHistory<Scalar>> solution_history,
    Tempus::Status&)
{
    // Get the working state.
    auto working_state = solution_history->getWorkingState();

    // Get minimum time step that ensures CFL <= 1
    _response_manager.evaluateResponses(working_state->getX(),
                                        working_state->getXDot(),
                                        working_state->getTime(),
                                        working_state->getTimeStep());
    const double dt_cfl1 = _response_manager.value();

    // Compute linear weight based on input parameters
    double wt;
    if (_cfl_transition > 0)
    {
        // Current time step index/time
        double dt_index = 0.0;
        if (_cfl_type == CflTransitionType::steps)
            dt_index = static_cast<double>(solution_history->getCurrentIndex());
        else
            dt_index = solution_history->getCurrentTime();

        if (dt_index < _cfl_transition_init)
        {
            wt = 0.0;
        }
        else if (dt_index >= _cfl_transition_init
                 && dt_index <= _cfl_transition + _cfl_transition_init)
        {
            wt = std::min((dt_index - _cfl_transition_init) / _cfl_transition,
                          1.0);
        }
        else
        {
            wt = 1.0;
        }
    }
    else
    {
        wt = 1.0;
    }

    // Compute time step based on desired CFL
    const double cfl_desired = (1.0 - wt) * _cfl_init + wt * _cfl;
    double dt = cfl_desired * dt_cfl1;

    // If it is out of the bounds specified by the user then restrict it
    // before setting it.
    const int dt_index = working_state->getIndex() - 1;
    if (dt_index == 0 && tsc.getInitTimeStep() != tsc.getMinTimeStep())
    {
        dt = tsc.getInitTimeStep();
    }
    else if (dt < tsc.getMinTimeStep())
    {
        dt = tsc.getMinTimeStep();
    }
    else if (dt > tsc.getMaxTimeStep())
    {
        dt = tsc.getMaxTimeStep();
    }
    working_state->setTimeStep(dt);
    working_state->setTime(solution_history->getCurrentTime() + dt);

    // Save current CFL so it may be accessed elsewhere
    this->setCurrentCFL(dt / dt_cfl1);
}

//---------------------------------------------------------------------------//

} // end namespace TempusTimeStepControl
} // end namespace VertexCFD

#endif // end VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALCFL_IMPL_HPP
