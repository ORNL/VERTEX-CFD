#ifndef VERTEXCFD_CLOSURE_VARIABLESUPGFLUX_HPP
#define VERTEXCFD_CLOSURE_VARIABLESUPGFLUX_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension SUPG flux evaluation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class VariableSUPGFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    VariableSUPGFlux(const panzer::IntegrationRule& ir,
                     const Teuchos::ParameterList& closure_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    std::string _variable_name;
    std::string _equation_name;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _var_diff_flux;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _tau_supg_var;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_var;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _var_source;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_var;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    enum TmpVars
    {
        VAR_RES,
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

#endif // end VERTEXCFD_CLOSURE_VARIABLESUPGFLUX_HPP
