#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFUNCTION_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFUNCTION_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension evaluation for LSVOF local entropy function term
//
// NOTE: Here we are saving each term within min function of eq. 16
// so were able to plot them.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFEntropyFunction
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFEntropyFunction(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& scalar_name,
        const std::string& equation_name);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _entropy_function;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _entropy_residual;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_scalar;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _scalar;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_scalar;

    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    enum EntropyTypes
    {
        quadratic,
        log,
        biquadratic
    };

    EntropyTypes _entropy_type;

    double _factor_entropy;

    enum TmpVars
    {
        TMP_SCALAR,
        TMP_LOG_SCALAR,
        NUM_TMPS
    };

    /// View type for shared memory
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFUNCTION_HPP
