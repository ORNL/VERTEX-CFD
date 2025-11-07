#ifndef VERTEXCFD_BOUNDARYSTATE_ELECTRICCURRENTDENSITYINSULATING_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_ELECTRICCURRENTDENSITYINSULATING_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
ElectricCurrentDensityInsulating<EvalType, Traits, NumSpaceDim>::
    ElectricCurrentDensityInsulating(const panzer::IntegrationRule& ir)
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

    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _boundary_velocity, "BOUNDARY_velocity_");
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _ext_magn_field, "external_magnetic_field_");

    this->setName("Boundary State Electric Current Density Insulating");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ElectricCurrentDensityInsulating<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
ElectricCurrentDensityInsulating<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _electric_potential.extent(1);
    const int num_grad_dim = _normals.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // using '_boundary_grad_electric_potential' as temporary variable
            // to calculate the U \cross B

            // x-component
            _boundary_grad_electric_potential(cell, point, 0)
                = _velocity[1](cell, point) * _ext_magn_field[2](cell, point);
            // y-component
            _boundary_grad_electric_potential(cell, point, 1)
                = -_velocity[0](cell, point) * _ext_magn_field[2](cell, point);
            if (num_grad_dim == 3)
            {
                // contribution to x-y from z-component
                _boundary_grad_electric_potential(cell, point, 0)
                    -= _velocity[2](cell, point)
                       * _ext_magn_field[1](cell, point);
                _boundary_grad_electric_potential(cell, point, 1)
                    += _velocity[2](cell, point)
                       * _ext_magn_field[0](cell, point);
                // z-component
                _boundary_grad_electric_potential(cell, point, 2)
                    = _velocity[0](cell, point)
                      * _ext_magn_field[1](cell, point);
                _boundary_grad_electric_potential(cell, point, 2)
                    -= _velocity[1](cell, point)
                       * _ext_magn_field[0](cell, point);
            }

            // Compute (\grad(\phi) - U \cross B) \cdot \vec{n}
            auto&& grad_phi_dot_n = _boundary_electric_potential(cell, point);
            grad_phi_dot_n = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                grad_phi_dot_n
                    += (_grad_electric_potential(cell, point, dim)
                        - _boundary_grad_electric_potential(cell, point, dim))
                       * _normals(cell, point, dim);
            }

            // using '_boundary_grad_electric_potential' as temporary variable
            // to calculate the (U \cross B) - (U_{bc} \cross B)

            // x-component
            _boundary_grad_electric_potential(cell, point, 0)
                -= _boundary_velocity[1](cell, point)
                   * _ext_magn_field[2](cell, point);
            // y-component
            _boundary_grad_electric_potential(cell, point, 1)
                -= -_boundary_velocity[0](cell, point)
                   * _ext_magn_field[2](cell, point);
            if (num_grad_dim == 3)
            {
                // contribution to x-y from z-component
                _boundary_grad_electric_potential(cell, point, 0)
                    += _boundary_velocity[2](cell, point)
                       * _ext_magn_field[1](cell, point);
                _boundary_grad_electric_potential(cell, point, 1)
                    -= _boundary_velocity[2](cell, point)
                       * _ext_magn_field[0](cell, point);
                // z-component
                _boundary_grad_electric_potential(cell, point, 2)
                    -= _boundary_velocity[0](cell, point)
                       * _ext_magn_field[1](cell, point);
                _boundary_grad_electric_potential(cell, point, 2)
                    += _boundary_velocity[1](cell, point)
                       * _ext_magn_field[0](cell, point);
            }

            // Set and boundary gradients
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                _boundary_grad_electric_potential(cell, point, dim)
                    += _grad_electric_potential(cell, point, dim)
                       - grad_phi_dot_n * _normals(cell, point, dim);
            }

            _boundary_electric_potential(cell, point)
                = _electric_potential(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_ELECTRICCURRENTDENSITYINSULATING_IMPL_HPP
