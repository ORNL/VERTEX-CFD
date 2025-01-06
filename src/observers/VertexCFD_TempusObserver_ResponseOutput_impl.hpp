#ifndef VERTEXCFD_TEMPUSOBSERVER_RESPONSEOUTPUT_IMPL_HPP
#define VERTEXCFD_TEMPUSOBSERVER_RESPONSEOUTPUT_IMPL_HPP

#include <iomanip>
#include <iostream>
#include <utility>

namespace VertexCFD
{
namespace TempusObserver
{
//---------------------------------------------------------------------------//
template<class Scalar>
ResponseOutput<Scalar>::ResponseOutput(
    Teuchos::RCP<Response::ResponseManager> response_manager,
    std::vector<int> output_freq)
    : _ostream(Teuchos::rcp(&std::cout, false))
    , _response_manager(response_manager)
    , _output_freq(std::move(output_freq))
{
    _ostream.setShowProcRank(false);
    _ostream.setOutputToRootOnly(0);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeStartIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    // When the initial time index is zero, this will output all responses.
    // Otherwise, output will depend on specified frequencies.
    outputResponses(integrator, integrator.getIndex());
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeStartTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeNextTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeBeforeTakeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeAfterTakeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeAfterCheckTimeStep(
    const Tempus::Integrator<Scalar>& /*integrator*/)
{
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeEndTimeStep(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Output responses at specified frequencies based on time step index.
    outputResponses(integrator, integrator.getIndex());
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::observeEndIntegrator(
    const Tempus::Integrator<Scalar>& integrator)
{
    // Output all responses unconditionally.
    outputResponses(integrator);
}

//---------------------------------------------------------------------------//
template<class Scalar>
void ResponseOutput<Scalar>::outputResponses(
    const Tempus::Integrator<Scalar>& integrator, const int current_index)
{
    const int num_resp = _response_manager->numResponses();

    // Just return if there are no responses to output.
    if (num_resp == 0)
        return;

    _response_manager->deactivateAll();

    // Activate desired responses.
    int num_outputs = 0;
    for (int i = 0; i < num_resp; ++i)
    {
        if (0 == current_index % _output_freq[i])
        {
            _response_manager->activateResponse(i);
            ++num_outputs;
        }
    }

    // Just return if there are no respones to output for this time step.
    if (num_outputs == 0)
        return;

    // Evaluate responses.
    const auto state = integrator.getSolutionHistory()->getCurrentState();
    _response_manager->evaluateResponses(state->getX(), state->getXDot(),
                                         state->getTime(), state->getTimeStep());

    // Outupt the integrated values.
    _ostream << "Scalar Responses:\n";
    for (int i = 0; i < num_resp; ++i)
    {
        if (0 == current_index % _output_freq[i])
        {
            const auto& name = _response_manager->name(i);
            const auto value = _response_manager->value(i);

            constexpr int prec = std::numeric_limits<double>::digits10 + 1;

            _ostream << "  " << name << " = " << std::setprecision(prec)
                     << value << '\n';
        }
    }
}
//---------------------------------------------------------------------------//

} // namespace TempusObserver
} // namespace VertexCFD

#endif // VERTEXCFD_TEMPUSOBSERVER_RESPONSEOUTPUT_IMPL_HPP
