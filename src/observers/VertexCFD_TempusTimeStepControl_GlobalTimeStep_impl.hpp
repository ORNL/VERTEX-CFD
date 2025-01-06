#ifndef VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALTIMESTEP_IMPL_HPP
#define VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALTIMESTEP_IMPL_HPP

#include "VertexCFD_TempusTimeStepControl_GlobalTimeStep.hpp"
#include <algorithm>

namespace VertexCFD
{
namespace TempusTimeStepControl
{
//---------------------------------------------------------------------------//
template<class Scalar>
GlobalTimeStep<Scalar>::GlobalTimeStep(
    const Teuchos::ParameterList& user_params,
    Teuchos::RCP<PhysicsManager> physics_manager)
    : _dt_transition_steps(user_params.isType<int>("Time step transition "
                                                   "steps")
                               ? user_params.get<int>("Time step transition "
                                                      "steps")
                               : 0)
    , _response_manager(physics_manager)
{
    _response_manager.addMinValueResponse("global_cfl_time_step", "local_dt");

    // For the new Tempus::TimeStepControlStrategy interface, we need to
    // set a few base class member variables. In particular, incorrect
    // behavior may result if "stepType_" is not set to "Variable".
    this->stepType_ = "Variable";
    this->strategyType_ = "Global Time Step Strategy";
    this->name_ = "Global Time Step Strategy";
}

//---------------------------------------------------------------------------//
// Determine the time step size.
template<class Scalar>
void GlobalTimeStep<Scalar>::setNextTimeStep(
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

    // Get time step index (1-based) and compute linear weight
    const auto dt_index = working_state->getIndex() - 1;
    const auto wt
        = _dt_transition_steps > 0
              ? std::min(static_cast<double>(dt_index) / _dt_transition_steps,
                         1.0)
              : 1.0;

    const double dt_init
        = std::max(tsc.getMinTimeStep(), tsc.getInitTimeStep());
    const double dt_final = tsc.getMaxTimeStep();
    const double dt = (1.0 - wt) * dt_init + wt * dt_final;

    working_state->setTimeStep(dt);
    working_state->setTime(solution_history->getCurrentState()->getTime() + dt);

    // Save current CFL so it may be accessed elsewhere
    this->setCurrentCFL(dt / dt_cfl1);
}

//---------------------------------------------------------------------------//

} // namespace TempusTimeStepControl
} // namespace VertexCFD

#endif // VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALTIMESTEP_IMPL_HPP
