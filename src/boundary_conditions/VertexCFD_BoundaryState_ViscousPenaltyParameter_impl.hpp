#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_IMPL_HPP

#include "Panzer_Workset_Utilities.hpp"
#include <Panzer_HierarchicParallelism.hpp>

#include <cmath>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ViscousPenaltyParameter<EvalType, Traits>::ViscousPenaltyParameter(
    const panzer::IntegrationRule& ir,
    const panzer::PureBasis& basis,
    const std::string& dof_name,
    const double& penalty)
    : _penalty_param("viscous_penalty_parameter_" + dof_name, ir.dl_scalar)
    , _basis_name(basis.name())
    , _num_space_dim(ir.spatial_dimension)
    , _penalty(penalty)
{
    // Note: Default penalty is set in BoundaryFluxBase
    // Add evaluated field
    this->addEvaluatedField(_penalty_param);

    this->setName("Boundary State Viscous Penalty Parameter \"" + dof_name
                  + "\"" + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ViscousPenaltyParameter<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ViscousPenaltyParameter<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _ip_gradients = this->wda(workset).bases[_basis_index]->grad_basis;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ViscousPenaltyParameter<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _ip_gradients.extent(1);
    const int num_point = _ip_gradients.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::sqrt;
            double one_over_h = 0.0;
            for (int basis = 0; basis < num_basis; ++basis)
            {
                double one_over_h2_basis = 0.0;
                for (int dim = 0; dim < _num_space_dim; ++dim)
                {
                    one_over_h2_basis
                        += _ip_gradients(cell, basis, point, dim)
                           * _ip_gradients(cell, basis, point, dim);
                }
                one_over_h += sqrt(one_over_h2_basis);
            }

            _penalty_param(cell, point) = _penalty * one_over_h;
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_IMPL_HPP
