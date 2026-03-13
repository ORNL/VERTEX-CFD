#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSGRADIENT_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSGRADIENT_HPP

#include "Panzer_PureBasis.hpp"
#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ViscousGradient : public panzer::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ViscousGradient(const panzer::IntegrationRule& ir,
                    const std::string& dof_name,
                    const double penalty_factor);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _grad;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _scaled_grad;

  private:
    int _num_space_dim;
    double _penalty_factor;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dof;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _bnd_dof;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _penalty_param;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSGRADIENT_HPP
