#ifndef VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Conduction exact solution.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class ConductionExactSolution : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    ConductionExactSolution(const panzer::IntegrationRule& ir,
                            const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<double, panzer::Cell, panzer::Point> _lagrange_pressure;
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        _velocity;
    PHX::MDField<double, panzer::Cell, panzer::Point> _temperature;

  private:
    int _ir_degree;
    double _q;
    double _k;
    double _T_right;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
};

//---------------------------------------------------------------------------//

} // namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_HPP
