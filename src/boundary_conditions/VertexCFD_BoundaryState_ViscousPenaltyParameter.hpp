#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_HPP

#include "Panzer_PureBasis.hpp"
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
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
class ViscousPenaltyParameter : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    ViscousPenaltyParameter(const panzer::IntegrationRule& ir,
                            const panzer::PureBasis& basis,
                            const std::string& dof_name,
                            const double& penalty);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _penalty_param;

  private:
    std::string _basis_name;
    int _num_space_dim;
    int _basis_index;
    double _penalty;

    PHX::MDField<double, panzer::Cell, panzer::BASIS, panzer::Point, panzer::Dim>
        _ip_gradients;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_HPP
