#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSGRADIENT_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSGRADIENT_IMPL_HPP

#include "Panzer_Workset_Utilities.hpp"
#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ViscousGradient<EvalType, Traits>::ViscousGradient(
    const panzer::IntegrationRule& ir, const std::string& dof_name)
    : _grad("PENALTY_GRAD_" + dof_name, ir.dl_vector)
    , _scaled_grad("SYMMETRY_GRAD_" + dof_name, ir.dl_vector)
    , _num_space_dim(ir.spatial_dimension)
    , _dof(dof_name, ir.dl_scalar)
    , _bnd_dof("BOUNDARY_" + dof_name, ir.dl_scalar)
    , _penalty_param("viscous_penalty_parameter_" + dof_name, ir.dl_scalar)
    , _normals("Side Normal", ir.dl_vector)
{
    // Add evaluated fields
    this->addEvaluatedField(_grad);
    this->addEvaluatedField(_scaled_grad);

    // Add dependent fields
    this->addDependentField(_dof);
    this->addDependentField(_bnd_dof);
    this->addDependentField(_penalty_param);
    this->addDependentField(_normals);

    this->setName("Boundary State Viscous Gradient "
                  + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ViscousGradient<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void ViscousGradient<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dof.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const auto u = _dof(cell, point);
            const auto u_bnd = _bnd_dof(cell, point);
            const auto delta = _penalty_param(cell, point);

            for (int dim = 0; dim < _num_space_dim; ++dim)
            {
                _grad(cell, point, dim) = _normals(cell, point, dim)
                                          * (u - u_bnd);

                _scaled_grad(cell, point, dim) = delta
                                                 * _grad(cell, point, dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSGRADIENT_IMPL_HPP
