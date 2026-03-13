#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSSIGN_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSSIGN_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
// Smoothed sign function for conservative level set method. See Quezada de
// Luna (2019). Generates sign fields for the current phi value, as well as the
// value at the end of the previous stage or time step.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class IncompressibleCLSSign : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr auto pi = Kokkos::numbers::pi_v<float>;

    IncompressibleCLSSign(const panzer::IntegrationRule& ir,
                          const std::string& field_prefix = "",
                          const bool& eval_star_sign = true);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _heaviside;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sign;
    PHX::MDField<double, panzer::Cell, panzer::Point> _sign_star;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _phi;
    PHX::MDField<const double, panzer::Cell, panzer::Point> _phi_star;
    PHX::MDField<const double, panzer::Cell, panzer::Point> _epsilon;
    bool _eval_star_sign;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSSIGN_HPP
