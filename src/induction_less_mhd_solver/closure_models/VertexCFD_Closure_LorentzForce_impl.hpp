#ifndef VERTEXCFD_CLOSURE_LORENTZFORCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_LORENTZFORCE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
LorentzForce<EvalType, Traits, NumSpaceDim>::LorentzForce(
    const panzer::IntegrationRule& ir)
    : _sigma("electrical_conductivity", ir.dl_scalar)
    , _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
{
    // Evaluated fields
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _lorentz_force, "VOLUMETRIC_SOURCE_momentum_");

    // Dependent fields
    this->addDependentField(_sigma);
    this->addDependentField(_grad_electric_potential);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _ext_magn_field, "external_magnetic_field_");

    this->setName("Electric Potential Lorentz Force "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void LorentzForce<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries in parallel for
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_sigma.get_view());
        bytes = scratch_view::shmem_size(_sigma.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_sigma.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
LorentzForce<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_electric_potential.extent(1);
    const int num_grad_dim = _grad_electric_potential.extent(2);

    // Initialize scratch memory
    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_sigma.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Cross product term
            _lorentz_force[0](cell, point)
                = -_grad_electric_potential(cell, point, 1)
                  * _ext_magn_field[2](cell, point);

            _lorentz_force[1](cell, point)
                = _grad_electric_potential(cell, point, 0)
                  * _ext_magn_field[2](cell, point);

            if (num_grad_dim == 3)
            {
                // x-component
                _lorentz_force[0](cell, point)
                    += _grad_electric_potential(cell, point, 2)
                       * _ext_magn_field[1](cell, point);
                // y-component
                _lorentz_force[1](cell, point)
                    -= _grad_electric_potential(cell, point, 2)
                       * _ext_magn_field[0](cell, point);
                // z-component
                _lorentz_force[2](cell, point)
                    = -_grad_electric_potential(cell, point, 0)
                          * _ext_magn_field[1](cell, point)
                      + _grad_electric_potential(cell, point, 1)
                            * _ext_magn_field[0](cell, point);
            }

            // Compute B norm and \vec{u} \cdot \vec{B}. NOTE: the external
            // magnetic field is assumed of the same dimension as the mesh in
            // this implementation.
            auto&& b2 = tmp_field(point, B2);
            auto&& b_dot_u = tmp_field(point, B_DOT_U);
            b2 = 0.0;
            b_dot_u = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                b2 += _ext_magn_field[dim](cell, point)
                      * _ext_magn_field[dim](cell, point);
                b_dot_u += _ext_magn_field[dim](cell, point)
                           * _velocity[dim](cell, point);
            }

            // Adding dot product terms
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                _lorentz_force[dim](cell, point)
                    += _ext_magn_field[dim](cell, point) * b_dot_u;
                _lorentz_force[dim](cell, point)
                    -= b2 * _velocity[dim](cell, point);
                _lorentz_force[dim](cell, point) *= _sigma(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_LORENTZFORCE_IMPL_HPP
