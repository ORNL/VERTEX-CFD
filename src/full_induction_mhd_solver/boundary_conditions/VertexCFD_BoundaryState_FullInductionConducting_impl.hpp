#ifndef VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONCONDUCTING_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONCONDUCTING_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// This function sets the boundary induced magnetic field and its gradient
// for a conducting wall with a specified boundary velocity. The calculation
// of the boundary gradient on a moving wall for resistive MHD assumes a
// constant resistivity, and may require updating for the case of variable
// resistivity.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
FullInductionConducting<EvalType, Traits, NumSpaceDim>::FullInductionConducting(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& bc_params,
    const MHDProperties::FullInductionMHDProperties& mhd_props)
    : _boundary_scalar_magnetic_potential("BOUNDARY_scalar_magnetic_potential",
                                          ir.dl_scalar)
    , _build_magn_corr(mhd_props.buildMagnCorr())
    , _build_resistive_flux(mhd_props.buildResistiveFlux())
    , _dirichlet_scalar_magn_pot(bc_params.isType<double>("scalar_magnetic_"
                                                          "potential"))
    , _bnd_scalar_magn_pot(std::numeric_limits<double>::signaling_NaN())
    , _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
    , _normals("Side Normal", ir.dl_vector)
    , _scalar_magnetic_potential("scalar_magnetic_potential", ir.dl_scalar)
    , _resistivity("resistivity", ir.dl_scalar)
{
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        const std::string magn_string = "induced_magnetic_field_"
                                        + std::to_string(dim);
        _bnd_magn_field[dim] = bc_params.get<double>(magn_string);
    }

    // Add evaluated/dependent fields
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _boundary_induced_magnetic_field,
                                   "BOUNDARY_induced_magnetic_field_");

    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_vector,
                                   _boundary_grad_induced_magnetic_field,
                                   "BOUNDARY_GRAD_induced_magnetic_field_");

    Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _induced_magnetic_field,
                                   "induced_magnetic_field_");

    Utils::addDependentVectorField(*this,
                                   ir.dl_vector,
                                   _grad_induced_magnetic_field,
                                   "GRAD_induced_magnetic_field_");

    if (_build_magn_corr)
    {
        this->addEvaluatedField(_boundary_scalar_magnetic_potential);
        if (_dirichlet_scalar_magn_pot)
        {
            _bnd_scalar_magn_pot
                = bc_params.get<double>("scalar_magnetic_potential");
        }
        else
        {
            this->addDependentField(_scalar_magnetic_potential);
        }
    }
    this->addDependentField(_normals);

    if (_build_resistive_flux)
    {
        this->addDependentField(_resistivity);
        Utils::addDependentVectorField(
            *this, ir.dl_scalar, _boundary_velocity, "BOUNDARY_velocity_");
        Utils::addDependentVectorField(*this,
                                       ir.dl_scalar,
                                       _external_magnetic_field,
                                       "external_magnetic_field_");
    }

    this->setName("Boundary State Full Induction Conducting "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void FullInductionConducting<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
FullInductionConducting<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_induced_magnetic_field[0].extent(1);
    const int num_grad_dim = _boundary_grad_induced_magnetic_field[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Compute B \cdot n
            scalar_type B_dot_n = 0.0;
            Kokkos::Array<scalar_type, num_space_dim> gradB_dot_n = {0};
            for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
            {
                B_dot_n += (_induced_magnetic_field[grad_dim](cell, point)
                            - _bnd_magn_field[grad_dim])
                           * _normals(cell, point, grad_dim);
                for (int field_dim = 0; field_dim < num_grad_dim; ++field_dim)
                {
                    gradB_dot_n[field_dim]
                        += _grad_induced_magnetic_field[field_dim](
                               cell, point, grad_dim)
                           * _normals(cell, point, grad_dim);
                }
            }

            // Compute the boundary induced magnetic field
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _boundary_induced_magnetic_field[dim](cell, point)
                    = _induced_magnetic_field[dim](cell, point);
                if (dim < num_grad_dim)
                {
                    _boundary_induced_magnetic_field[dim](cell, point)
                        -= B_dot_n * _normals(cell, point, dim);
                }
            }

            if (_build_magn_corr)
            {
                if (_dirichlet_scalar_magn_pot)
                {
                    _boundary_scalar_magnetic_potential(cell, point)
                        = _bnd_scalar_magn_pot;
                }
                else
                {
                    _boundary_scalar_magnetic_potential(cell, point)
                        = _scalar_magnetic_potential(cell, point);
                }
            }

            // include resistive contributions to boundary gradient from moving
            // wall
            if (_build_resistive_flux)
            {
                const scalar_type inv_eta = _magnetic_permeability
                                            / _resistivity(cell, point);
                for (int d = 0; d < num_grad_dim; ++d)
                {
                    for (int fdim = 0; fdim < num_grad_dim; ++fdim)
                    {
                        if (d == fdim)
                            continue;
                        gradB_dot_n[fdim]
                            += _normals(cell, point, d) * inv_eta
                               * (_boundary_velocity[fdim](cell, point)
                                      * (_external_magnetic_field[d](cell, point)
                                         + _bnd_magn_field[d])
                                  - _boundary_velocity[d](cell, point)
                                        * (_external_magnetic_field[fdim](
                                               cell, point)
                                           + _bnd_magn_field[fdim]));
                    }
                }
            }

            // Set gradients
            for (int d = 0; d < num_grad_dim; ++d)
            {
                for (int fdim = 0; fdim < num_space_dim; ++fdim)
                {
                    _boundary_grad_induced_magnetic_field[fdim](cell, point, d)
                        = _grad_induced_magnetic_field[fdim](cell, point, d);
                    if (fdim < num_grad_dim)
                    {
                        _boundary_grad_induced_magnetic_field[fdim](
                            cell, point, d)
                            -= gradB_dot_n[fdim] * _normals(cell, point, d);
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONCONDUCTING_IMPL_HPP
