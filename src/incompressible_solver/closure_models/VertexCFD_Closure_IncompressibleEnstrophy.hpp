#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEENSTROPHY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEENSTROPHY_HPP

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
// Calculates vorticity vector and enstrophy magnitude
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleEnstrophy : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleEnstrophy(const panzer::IntegrationRule& ir);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _vorticity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _enstrophy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _divergence;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _mag_divergence;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEENSTROPHY_HPP
