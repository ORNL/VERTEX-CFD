#ifndef VERTEXCFD_CLOSURE_HARTMANNPROBLEMEXACT_HPP
#define VERTEXCFD_CLOSURE_HARTMANNPROBLEMEXACT_HPP

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
// Hartmann Problem Exact Solution
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class HartmannProblemExact : public panzer::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    HartmannProblemExact(const panzer::IntegrationRule& ir,
                         const Teuchos::ParameterList& closure_params,
                         const Teuchos::ParameterList& user_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _exact_lagrange_pressure;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _exact_velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _exact_elec_pot;

  private:
    double _nu;
    double _sigma;
    double _rho;
    double _L;
    double _B;
    double _Ha;
    int _ir_degree;
    int _ir_index;

    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim> _ip_coords;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_HARTMANNPROBLEMEXACT_HPP
