#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ViscousPenaltyParameter<EvalType, Traits>::ViscousPenaltyParameter(
    const panzer::IntegrationRule& ir, const panzer::PureBasis& basis)
    : _penalty_param("viscous_penalty_parameter", ir.dl_scalar)
    , _basis_name(basis.name())
    , _num_space_dim(ir.spatial_dimension)
{
    // Note: Default penalty is set in BoundaryFluxBase
    // Add evaluated field
    this->addEvaluatedField(_penalty_param);

    this->setName("Boundary State Viscous Penalty Parameter in "
                  + std::to_string(_num_space_dim) + "D");
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

    // Allocate space for thread-local temporaries in parallel for
    const size_t bytes
        = scratch_view::shmem_size(_penalty_param.extent(1), NUM_TMPS);

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
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

    // Initialize scratch memory
    scratch_view tmp_field
        = scratch_view(team.team_shmem(), num_point, NUM_TMPS);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using Kokkos::sqrt;
            _penalty_param(cell, point) = 0.0;
            for (int basis = 0; basis < num_basis; ++basis)
            {
                auto&& one_over_h2_basis = tmp_field(point, ONE_OVER_H2);
                one_over_h2_basis = 0.0;
                for (int dim = 0; dim < _num_space_dim; ++dim)
                {
                    one_over_h2_basis
                        += _ip_gradients(cell, basis, point, dim)
                           * _ip_gradients(cell, basis, point, dim);
                }
                _penalty_param(cell, point) += sqrt(one_over_h2_basis);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETER_IMPL_HPP
