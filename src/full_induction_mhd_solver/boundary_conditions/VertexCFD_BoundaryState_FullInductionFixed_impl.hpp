#ifndef VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONFIXED_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONFIXED_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// This function sets fixed boundary values for the induced magnetic field
// components from input. When solving the magnetic correction potential
// equation for divergence cleaning, the boundary value for the scalar
// magnetic potential is extrapolated (free) unless a value is set in the
// input BC parameters.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
FullInductionFixed<EvalType, Traits, NumSpaceDim>::FullInductionFixed(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& bc_params,
    const MHDProperties::FullInductionMHDProperties& mhd_props)
    : _boundary_scalar_magnetic_potential("BOUNDARY_scalar_magnetic_potential",
                                          ir.dl_scalar)
    , _build_magn_corr(mhd_props.buildMagnCorr())
    , _dirichlet_scalar_magn_pot(bc_params.isType<double>("scalar_magnetic_"
                                                          "potential"))
    , _bnd_scalar_magn_pot(std::numeric_limits<double>::signaling_NaN())
    , _scalar_magnetic_potential("scalar_magnetic_potential", ir.dl_scalar)
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

    this->setName("Boundary State Full Induction Fixed "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void FullInductionFixed<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
FullInductionFixed<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_induced_magnetic_field[0].extent(1);
    const int num_grad_dim = _boundary_grad_induced_magnetic_field[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set the fixed boundary values
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _boundary_induced_magnetic_field[dim](cell, point)
                    = _bnd_magn_field[dim];
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

            // Set gradients
            for (int d = 0; d < num_grad_dim; ++d)
            {
                for (int field_dim = 0; field_dim < num_space_dim; ++field_dim)
                {
                    _boundary_grad_induced_magnetic_field[field_dim](
                        cell, point, d)
                        = _grad_induced_magnetic_field[field_dim](
                            cell, point, d);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONFIXED_IMPL_HPP
