#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARTIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARTIMEDERIVATIVE_HPP

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
// Time derivative evaluation for LSVOF scalar transport equations
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleLSVOFScalarTimeDerivative
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    IncompressibleLSVOFScalarTimeDerivative(const panzer::IntegrationRule& ir,
                                            const std::string& dof_name,
                                            const std::string& eqn_name,
                                            const bool& mass_weighted = false);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_dof;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dof;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_dof;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_rho;

    bool _mass_weighted;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARTIMEDERIVATIVE_HPP
