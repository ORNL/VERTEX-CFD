#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_MACHINELEARNING_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_MACHINELEARNING_HPP

#ifdef VERTEXCFD_HAVE_TENSORFLOW

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension convective flux evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleConvectiveFluxMachineLearning
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleConvectiveFluxMachineLearning(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& fluid_params,
        const Teuchos::ParameterList& closure_params,
        const std::string& flux_prefix = "",
        const std::string& field_prefix = "");

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    struct InputStep
    {
    };
    struct OutputStep
    {
    };

    KOKKOS_INLINE_FUNCTION void
    operator()(InputStep, int cell, int point) const;

    KOKKOS_INLINE_FUNCTION void
    operator()(OutputStep, int cell, int point) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _continuity_flux;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _energy_flux;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _ml_velocity;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _momentum_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    double _tf_input_scale;
    double _tf_input_mean;
    double _tf_input_range;
    double _tf_output_scale;
    double _tf_output_mean;
    double _tf_output_range;

    double _rho;
    double _rhoCp;
    double _h_min;
    double _h_max;
    double _U_max;

    int _ir_degree;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;

    std::shared_ptr<tflite::Interpreter> _interpreter_shared_velocity;
    std::shared_ptr<tflite::FlatBufferModel> _model_velocity;
    int _input_id_velocity;
    Kokkos::View<float*, PHX::Device> _device_view_input_tensor_velocity;
    Kokkos::View<float*, PHX::Device> _device_view_output_tensor_velocity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_HAVE_TENSORFLOW
#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECONVECTIVEFLUX_MACHINELEARNING_HPP
