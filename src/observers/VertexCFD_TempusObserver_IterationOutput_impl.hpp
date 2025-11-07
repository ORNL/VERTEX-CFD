#ifndef VERTEXCFD_TEMPUSOBSERVER_ITERATIONOUTPUT_IMPL_HPP
#define VERTEXCFD_TEMPUSOBSERVER_ITERATIONOUTPUT_IMPL_HPP

#include <cmath>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
IterationOutput<Scalar>::IterationOutput(
    Teuchos::RCP<TempusTimeStepControl::Strategy<Scalar>> dt_strategy)
    : _ostream(Teuchos::rcp(&std::cout, false))
    , _dt_strategy(dt_strategy)
{
    _ostream.setShowProcRank(false);
    _ostream.setOutputToRootOnly(0);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeStartIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    const std::time_t begin = std::time(nullptr);
    _ostream << "\n==========================================================="
                "=================\n"
             << "Time Integration Begin\n"
             << std::asctime(std::localtime(&begin))
             << "\n  Stepper = " << integrator.getStepper()->description()
             << "\n  Simulation Time Range  ["
             << integrator.getTimeStepControl()->getInitTime() << ", "
             << integrator.getTimeStepControl()->getFinalTime() << "]"
             << "\n-----------------------------------------------------------"
                "-----------------\n";

    auto tsc = integrator.getTimeStepControl();
    const auto final_time = tsc->getFinalTime();
    const auto dt_min = tsc->getMinTimeStep();

    // Get location of first significant digit of a double
    auto get_digit = [](double t) -> int {
        return t > 0.0 ? std::floor(std::log10(t)) : 0;
    };

    const int t_digit = get_digit(final_time);
    const int dt_digit = get_digit(dt_min);

    // Make sure we always get meaningful time precision
    _time_precision = t_digit - dt_digit + 2;
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeStartTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeNextTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeBeforeTakeStep(
    const Tempus::Integrator<Scalar>& integrator)
{
    const auto current_time = integrator.getTime();
    const auto state = integrator.getSolutionHistory()->getWorkingState();
    _ostream << "\nTime Step = " << state->getIndex();
    _ostream << "; Order = " << state->getOrder();
    _ostream << "\nCFL = " << std::setprecision(3) << std::scientific
             << _dt_strategy->currentCFL();
    _ostream << "; dt = " << std::setprecision(3) << std::scientific
             << state->getTimeStep();
    _ostream << "; Time = " << std::setprecision(_time_precision)
             << std::scientific << current_time << "\n";
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeAfterTakeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeAfterCheckTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeEndTimeStep(
    const Tempus::Integrator<Scalar>& integrator)
{
    const auto stepper_time = integrator.getStepperTimer()->totalElapsedTime();
    integrator.getStepperTimer()->reset();
    _ostream << "Time step time to completion (s): " << std::setprecision(2)
             << std::scientific << stepper_time;
    _ostream << "\n";
}

//---------------------------------------------------------------------------//
template<class Scalar>
void IterationOutput<Scalar>::observeEndIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    std::string exit_status;
    if (integrator.getSolutionHistory()->getCurrentState()->getSolutionStatus()
            == Tempus::Status::FAILED
        or integrator.getStatus() == Tempus::Status::FAILED)
    {
        exit_status = "Time integration FAILURE!";
    }
    else
    {
        exit_status = "Time integration complete.";
    }
    const std::time_t end = std::time(nullptr);
    const auto runtime_sec
        = integrator.getIntegratorTimer()->totalElapsedTime();
    const auto runtime_min = runtime_sec / 60;
    const auto runtime_hr = runtime_min / 60;
    _ostream << "\n-----------------------------------------------------------"
                "-----------------\n"
             << "Total runtime = " << std::setprecision(2) << std::scientific
             << runtime_sec << " sec\n"
             << "              = " << std::setprecision(2) << std::scientific
             << runtime_min << " min\n"
             << "              = " << std::setprecision(2) << std::fixed
             << runtime_hr << " hr\n"
             << std::asctime(std::localtime(&end)) << exit_status
             << "\n==========================================================="
                "=================\n"
             << "\n";
}

//---------------------------------------------------------------------------//

} // end namespace TempusObserver
} // end namespace VertexCFD

#endif // end VERTEXCFD_TEMPUSOBSERVER_ITERATIONOUTPUT_IMPL_HPP
