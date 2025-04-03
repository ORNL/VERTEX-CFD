#ifndef VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALINSULATINGWALL_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALINSULATINGWALL_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ElectricPotentialInsulatingWall<EvalType, Traits>::ElectricPotentialInsulatingWall(
    const panzer::IntegrationRule& ir)
    : _boundary_electric_potential("BOUNDARY_electric_potential", ir.dl_scalar)
    , _boundary_grad_electric_potential("BOUNDARY_GRAD_electric_potential",
                                        ir.dl_vector)
    , _electric_potential("electric_potential", ir.dl_scalar)
    , _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
{
    this->addDependentField(_electric_potential);
    this->addEvaluatedField(_boundary_electric_potential);
    this->addDependentField(_grad_electric_potential);
    this->addEvaluatedField(_boundary_grad_electric_potential);

    this->addDependentField(_normals);

    this->setName("Boundary State Electric Potential Insulating Wall");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ElectricPotentialInsulatingWall<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ElectricPotentialInsulatingWall<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _electric_potential.extent(1);
    const int num_grad_dim = _normals.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Compute \grad(\phi) \cdot \vec{n}
            scalar_type grad_phi_dot_n = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                grad_phi_dot_n += _grad_electric_potential(cell, point, dim)
                                  * _normals(cell, point, dim);
            }

            _boundary_electric_potential(cell, point)
                = _electric_potential(cell, point);

            // Set and boundary gradients
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                _boundary_grad_electric_potential(cell, point, dim)
                    = _grad_electric_potential(cell, point, dim)
                      - grad_phi_dot_n * _normals(cell, point, dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALINSULATINGWALL_IMPL_HPP
