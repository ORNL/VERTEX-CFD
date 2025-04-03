#ifndef VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALCFL_HPP
#define VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALCFL_HPP

#include "observers/VertexCFD_TempusTimeStepControl_Strategy.hpp"

#include "drivers/VertexCFD_PhysicsManager.hpp"
#include "responses/VertexCFD_ResponseManager.hpp"

#include <Tempus_SolutionHistory.hpp>
#include <Tempus_SolutionState.hpp>
#include <Tempus_StepperState.hpp>
#include <Tempus_TimeStepControl.hpp>
#include <Tempus_TimeStepControlStrategy.hpp>

#include <Thyra_VectorBase.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace TempusTimeStepControl
{
//---------------------------------------------------------------------------//
template<class Scalar>
class GlobalCFL : virtual public Strategy<Scalar>
{
  public:
    GlobalCFL(const Teuchos::ParameterList& user_params,
              Teuchos::RCP<PhysicsManager> physics_manager);

    // Determine the time step size.
    void setNextTimeStep(
        const Tempus::TimeStepControl<Scalar>& tsc,
        Teuchos::RCP<Tempus::SolutionHistory<Scalar>> solution_history,
        Tempus::Status& integrator_status) override;

  private:
    enum CflTransitionType
    {
        steps,
        time
    };

    double _cfl;
    double _cfl_init;
    double _cfl_transition_init;
    double _cfl_transition;
    CflTransitionType _cfl_type;
    Response::ResponseManager _response_manager;
};

//---------------------------------------------------------------------------//

} // end namespace TempusTimeStepControl
} // end namespace VertexCFD

#include "VertexCFD_TempusTimeStepControl_GlobalCFL_impl.hpp"

#endif // end VERTEXCFD_TEMPUSTIMESTEPCONTROL_GLOBALCFL_HPP
