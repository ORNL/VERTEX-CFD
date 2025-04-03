#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// Boundary condition for wall-resolved K-Omega simulations with ramping
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
TurbulenceKOmegaWallResolved<EvalType, Traits>::TurbulenceKOmegaWallResolved(
    const panzer::IntegrationRule& ir, const Teuchos::ParameterList& bc_params)
    : _boundary_k("BOUNDARY_turb_kinetic_energy", ir.dl_scalar)
    , _boundary_w("BOUNDARY_turb_specific_dissipation_rate", ir.dl_scalar)
    , _boundary_grad_k("BOUNDARY_GRAD_turb_kinetic_energy", ir.dl_vector)
    , _boundary_grad_w("BOUNDARY_GRAD_turb_specific_dissipation_rate",
                       ir.dl_vector)
    , _k("turb_kinetic_energy", ir.dl_scalar)
    , _grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
    , _grad_w("GRAD_turb_specific_dissipation_rate", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _omega_wall(bc_params.get<double>("Omega Wall Value"))
    , _omega_wall_init(_omega_wall)
    , _omega_ramp_time(0.0)
    , _num_grad_dim(ir.spatial_dimension)
{
    // Check for ramping parameters
    if (bc_params.isType<double>("Omega Wall Initial Value"))
    {
        _omega_wall_init = bc_params.get<double>("Omega Wall Initial Value");
        _omega_ramp_time = bc_params.get<double>("Omega Ramp Time");
    }

    // Add evaluated fields
    this->addEvaluatedField(_boundary_k);
    this->addEvaluatedField(_boundary_w);
    this->addEvaluatedField(_boundary_grad_k);
    this->addEvaluatedField(_boundary_grad_w);

    // Add dependent fields
    this->addDependentField(_k);
    this->addDependentField(_grad_k);
    this->addDependentField(_grad_w);
    this->addDependentField(_normals);

    this->setName("Boundary State Turbulence Model KOmegaWallResolved "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void TurbulenceKOmegaWallResolved<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _time = workset.time;

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
TurbulenceKOmegaWallResolved<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_k.extent(1);
    using std::pow;

    double omega_wall = _omega_wall;

    // Ramp omega wall value using logarithmic interpolation
    if (_time < _omega_ramp_time)
    {
        omega_wall = pow(_omega_wall_init, 1.0 - _time / _omega_ramp_time)
                     * pow(_omega_wall, _time / _omega_ramp_time);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Assign boundary values
            _boundary_k(cell, point) = _k(cell, point);
            _boundary_w(cell, point) = omega_wall;

            // Assign boundary gradients
            for (int d = 0; d < _num_grad_dim; ++d)
            {
                _boundary_grad_k(cell, point, d) = _grad_k(cell, point, d);
                _boundary_grad_w(cell, point, d) = _grad_w(cell, point, d);
            }

            // Subtract wall normal component from grad(k)
            for (int d = 0; d < _num_grad_dim; ++d)
            {
                for (int grad_dim = 0; grad_dim < _num_grad_dim; ++grad_dim)
                {
                    _boundary_grad_k(cell, point, d)
                        -= _grad_k(cell, point, grad_dim)
                           * _normals(cell, point, grad_dim)
                           * _normals(cell, point, d);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_IMPL_HPP
