#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_MACHINELEARNING_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_MACHINELEARNING_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/logger.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <Teuchos_YamlParser_decl.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleConvectiveFluxMachineLearning<EvalType, Traits, NumSpaceDim>::
    IncompressibleConvectiveFluxMachineLearning(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& closure_params,
        const std::string& flux_prefix,
        const std::string& field_prefix)
    : _continuity_flux(flux_prefix + "CONVECTIVE_FLUX_continuity", ir.dl_vector)
    , _energy_flux(flux_prefix + "CONVECTIVE_FLUX_energy", ir.dl_vector)
    , _ml_velocity("ML_velocity", ir.dl_scalar)
    , _lagrange_pressure(field_prefix + "lagrange_pressure", ir.dl_scalar)
    , _temperature(field_prefix + "temperature", ir.dl_scalar)
    , _rho(closure_params.get<double>("Density"))
    , _rhoCp(_rho * closure_params.get<double>("Specific heat capacity"))
    , _ir_degree(ir.cubature_degree)
{
    if (!fluid_prop.solveTemperature())
        Kokkos::abort(
            "The IncompressibleConvectiveFluxMachineLearning model requires "
            "solving for the temperature!");

    // Evaluated fields
    this->addEvaluatedField(_continuity_flux);
    this->addEvaluatedField(_ml_velocity);

    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _momentum_flux,
                                 flux_prefix + "CONVECTIVE_FLUX_"
                                               "momentum_");

    this->addEvaluatedField(_energy_flux);

    // Dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");
    this->addDependentField(_lagrange_pressure);

    this->addDependentField(_temperature);

    // Get base file name
    const std::string tf_filename_velocity
        = closure_params.get<std::string>("TensorFlow File");

    const auto tf_metadata = Teuchos::YAMLParameterList::parseYamlFile(
        closure_params.get<std::string>("TensorFlow "
                                        "Metadata File"));

    const auto tf_metadata_normalization
        = tf_metadata->sublist("normalization");
    const auto tf_metadata_range = tf_metadata->sublist("range");

    _tf_input_scale = tf_metadata_normalization.sublist("input")
                          .sublist("coordinate")
                          .sublist("standard")
                          .get<double>("scale");
    _tf_input_mean = tf_metadata_normalization.sublist("input")
                         .sublist("coordinate")
                         .sublist("standard")
                         .get<double>("mean");
    _tf_input_range
        = tf_metadata_range.sublist("input").get<Teuchos::Array<double>>(
            "coordinate")[1];

    _tf_output_scale = tf_metadata_normalization.sublist("output")
                           .sublist("velocity")
                           .sublist("standard")
                           .get<double>("scale");
    _tf_output_mean = tf_metadata_normalization.sublist("output")
                          .sublist("velocity")
                          .sublist("standard")
                          .get<double>("mean");
    _tf_output_range
        = tf_metadata_range.sublist("output").get<Teuchos::Array<double>>(
            "velocity")[1];

    _h_min = closure_params.get<double>("H min");
    _h_max = closure_params.get<double>("H max");

    const double h = .5 * (_h_max - _h_min);
    const double nu = closure_params.get<double>("Kinematic viscosity");
    const double S_u = closure_params.get<double>("Momentum source");
    _U_max = 1.5 * S_u * h * h / 3.0 / nu;

    // By default TensorFlow about is very verbose
    tflite::LoggerOptions::SetMinimumLogSeverity(tflite::TFLITE_LOG_WARNING);

    // Setup tensorflow model from file for 1D-velocity profile of the second
    // component
    _model_velocity
        = tflite::FlatBufferModel::BuildFromFile(tf_filename_velocity.c_str());
    if (nullptr == _model_velocity)
    {
        throw std::runtime_error(
            "Flat buffer _model_velocity failed to build for file "
            + tf_filename_velocity);
    }

    // Build interpreters.
    const tflite::ops::builtin::BuiltinOpResolver resolver_velocity;
    tflite::InterpreterBuilder builder_velocity(*_model_velocity,
                                                resolver_velocity);

    std::unique_ptr<tflite::Interpreter> _interpreter_velocity;
    builder_velocity(&_interpreter_velocity);
    if (nullptr == _interpreter_velocity)
    {
        throw std::runtime_error("TensorFlow Intepreter failed to build.");
    }
    _input_id_velocity = _interpreter_velocity->inputs()[0];

    // Move to shared pointer
    _interpreter_shared_velocity = std::move(_interpreter_velocity);

    this->setName("Incompressible Convective Flux Machine Learning"
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleConvectiveFluxMachineLearning<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleConvectiveFluxMachineLearning<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    const int n_quadrature_points = _continuity_flux.extent(1);

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;

    // Build interpreters.
    _interpreter_shared_velocity->ResizeInputTensor(
        _input_id_velocity,
        {workset.num_cells * n_quadrature_points,
         /*inputs per quadrature point*/ 1});
    _interpreter_shared_velocity->AllocateTensors();

    if (_interpreter_shared_velocity->input_tensor(0)->bytes / sizeof(float)
        != static_cast<size_t>(workset.num_cells) * n_quadrature_points)
        throw std::runtime_error(
            "TensorFlow Interpreter has wrong input size.");

    const typename PHX::Device::memory_space memory_space;

    float* input_tensor_velocity
        = _interpreter_shared_velocity->typed_input_tensor<float>(
            _input_id_velocity);
    const Kokkos::View<float*, Kokkos::HostSpace> view_input_tensor_velocity(
        input_tensor_velocity, workset.num_cells * n_quadrature_points);
    _device_view_input_tensor_velocity
        = Kokkos::create_mirror_view(memory_space, view_input_tensor_velocity);

    const PHX::Device::execution_space exec;

    Kokkos::parallel_for(
        this->getName() + " input step",
        Kokkos::MDRangePolicy<PHX::Device, Kokkos::Rank<2>, InputStep>(
            exec, {0, 0}, {workset.num_cells, n_quadrature_points}),
        *this);
    exec.fence();

    Kokkos::deep_copy(view_input_tensor_velocity,
                      _device_view_input_tensor_velocity);

    if (_interpreter_shared_velocity->output_tensor(0)->bytes / sizeof(float)
        != static_cast<size_t>(workset.num_cells) * n_quadrature_points)
        throw std::runtime_error(
            "TensorFlow Interpreter has wrong output size.");

    if (_interpreter_shared_velocity->Invoke() != kTfLiteOk)
        Kokkos::abort("Failed to invoke model");

    float* output_tensor_velocity
        = _interpreter_shared_velocity->typed_output_tensor<float>(0);
    if (output_tensor_velocity == nullptr)
        Kokkos::abort("Output tensor is null");

    const int output_size
        = _interpreter_shared_velocity->output_tensor(0)->bytes / sizeof(float);
    if (output_size < 1)
        Kokkos::abort("Output tensor is empty");

    const Kokkos::View<float*, Kokkos::HostSpace> view_output_tensor_velocity(
        output_tensor_velocity, workset.num_cells * n_quadrature_points);

    _device_view_output_tensor_velocity = Kokkos::create_mirror_view_and_copy(
        memory_space, view_output_tensor_velocity);

    Kokkos::parallel_for(
        this->getName() + " output step",
        Kokkos::MDRangePolicy<PHX::Device, Kokkos::Rank<2>, OutputStep>(
            exec, {0, 0}, {workset.num_cells, n_quadrature_points}),
        *this);
    exec.fence();
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_FUNCTION void
IncompressibleConvectiveFluxMachineLearning<EvalType, Traits, NumSpaceDim>::
operator()(InputStep, int cell, int point) const
{
    const int n_quadrature_points = _continuity_flux.extent(1);
    // We assume that the model maps [-1,1] to [0,1].
    // Thus, to transform from [y_min, y_max] we have to apply
    // f(x) = 2*(x-y_min)/(y_max-y_min) - 1
    // = (2*(x-y_min)-y_max+y_min)/(y_max-y_min)
    // = (x - (y_min+y_max)/2)/((y_max-y_min)/2)
    const double H_mean = (_h_min + _h_max) / 2;
    const double H_range = (_h_max - _h_min) / 2;
    _device_view_input_tensor_velocity[cell * n_quadrature_points + point]
        = (((_ip_coords(cell, point, 1) - H_mean) / H_range) * _tf_input_range
           - _tf_input_mean)
          / _tf_input_scale;
}

template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleConvectiveFluxMachineLearning<EvalType, Traits, NumSpaceDim>::
operator()(OutputStep, int cell, int point) const
{
    const int n_quadrature_points = _continuity_flux.extent(1);
    Kokkos::Array<scalar_type, NumSpaceDim> velocity{};
    velocity[0] = (_device_view_output_tensor_velocity(
                       cell * n_quadrature_points + point)
                       * _tf_output_scale
                   + _tf_output_mean)
                  / _tf_output_range * _U_max;

    _ml_velocity(cell, point) = velocity[0];

    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        const scalar_type vel_dim = velocity[dim];
        _continuity_flux(cell, point, dim) = _rho * _velocity[dim](cell, point);

        for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
        {
            _momentum_flux[mom_dim](cell, point, dim)
                = _rho * _velocity[dim](cell, point)
                  * _velocity[mom_dim](cell, point);
            if (mom_dim == dim)
            {
                _momentum_flux[mom_dim](cell, point, dim)
                    += _lagrange_pressure(cell, point);
            }
        }

        _energy_flux(cell, point, dim) = _rhoCp * vel_dim
                                         * _temperature(cell, point);
    }
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_MACHINELEARNING_IMPL_HPP
