#ifndef VERTEXCFD_TEMPUSOBSERVER_ITERATIONOUTPUT_HPP
#define VERTEXCFD_TEMPUSOBSERVER_ITERATIONOUTPUT_HPP

#include "observers/VertexCFD_TempusTimeStepControl_Strategy.hpp"

#include <Tempus_IntegratorObserver.hpp>

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_STK_ResponseEvaluatorFactory_SolutionWriter.hpp>
#include <Panzer_STK_Utilities.hpp>

#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
class IterationOutput : virtual public Tempus::IntegratorObserver<Scalar>
{
  public:
    IterationOutput(
        Teuchos::RCP<TempusTimeStepControl::Strategy<Scalar>> dt_strategy);

    /// Observe the beginning of the time integrator.
    void observeStartIntegrator(
        const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe the beginning of the time step loop.
    void
    observeStartTimeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe after the next time step size is selected. The
    /// observer can choose to change the current integratorStatus.
    void
    observeNextTimeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe before Stepper takes step.
    void
    observeBeforeTakeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe after Stepper takes step.
    void
    observeAfterTakeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe after checking time step. Observer can still fail the time step
    /// here.
    void observeAfterCheckTimeStep(
        const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe the end of the time step loop.
    void
    observeEndTimeStep(const Tempus::Integrator<Scalar>& integrator) override;

    /// Observe the end of the time integrator.
    void
    observeEndIntegrator(const Tempus::Integrator<Scalar>& integrator) override;

  private:
    Teuchos::FancyOStream _ostream;
    Teuchos::RCP<TempusTimeStepControl::Strategy<Scalar>> _dt_strategy;
    int _time_precision = 3;
};

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#include "VertexCFD_TempusObserver_IterationOutput_impl.hpp"

#endif // end VERTEXCFD_TEMPUSOBSERVER_ITERATIONOUTPUT_HPP