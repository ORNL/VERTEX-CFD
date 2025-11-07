#ifndef VERTEXCFD_CLOSURE_RADLOCALTIMESTEPSIZE_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADLOCALTIMESTEPSIZE_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
RADLocalTimeStepSize<EvalType, Traits, NumSpaceDim>::RADLocalTimeStepSize(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop)
    : _num_species(species_prop.numSpecies())
    , _local_dt("local_dt", ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
    , _bateman_matrix(Kokkos::ViewAllocateWithoutInitializing("bateman_"
                                                              "matrix"),
                      _num_species,
                      _num_species)
    , _build_reaction(species_prop.buildReaction())
    , _build_advection(species_prop.buildAdvection())
    , _build_diffusion(species_prop.buildDiffusion())
{
    // Add evaluated field
    this->addEvaluatedField(_local_dt);

    // Add dependent fields
    if (_build_reaction)
    {
        // Copy bateman matrix from host to device
        auto bateman_matrix_host
            = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _bateman_matrix);
        bateman_matrix_host = species_prop.batemanMatrix();
        Kokkos::deep_copy(_bateman_matrix, bateman_matrix_host);
    }

    if (_build_advection)
        Utils::addDependentVectorField(
            *this, ir.dl_scalar, _velocity, "velocity_");

    if (_build_diffusion)
        _nu = species_prop.constantDiffusionCoefficient();

    if (_build_advection || _build_diffusion)
        this->addDependentField(_element_length);

    this->setName("RAD Local Time Step Size");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void RADLocalTimeStepSize<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_local_dt.get_view());
        bytes = scratch_view::shmem_size(
            _local_dt.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_local_dt.extent(1), NUM_TMPS);
    }
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
RADLocalTimeStepSize<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _local_dt.extent(1);
    const int num_grad_dim = _element_length.extent(2);
    using Kokkos::abs;
    using Kokkos::min;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_local_dt.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _local_dt(cell, point) = std::numeric_limits<double>::max();

            auto&& local_dt_react = tmp_field(point, LOCAL_DT_REACT);
            auto&& local_dt_adv = tmp_field(point, LOCAL_DT_ADV);
            auto&& local_dt_dif = tmp_field(point, LOCAL_DT_DIF);

            local_dt_react = 0.0;
            local_dt_adv = 0.0;
            local_dt_dif = 0.0;

            if (_build_reaction)
            {
                local_dt_react = 1.0 / abs(_bateman_matrix(0, 0));
                for (int num = 1; num < _num_species; ++num)
                {
                    local_dt_react = min(local_dt_react,
                                         1.0 / abs(_bateman_matrix(num, num)));
                }
                _local_dt(cell, point)
                    = min(_local_dt(cell, point), local_dt_react);
            }

            if (_build_advection)
            {
                // Accumulate 1/dt for velocity
                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    local_dt_adv += abs(_velocity[dim](cell, point))
                                    / _element_length(cell, point, dim);
                }

                // Take minimum time step
                local_dt_adv = 1.0 / local_dt_adv;
                _local_dt(cell, point)
                    = min(_local_dt(cell, point), local_dt_adv);
            }

            // Consider velocity if diffusion terms are included
            if (_build_diffusion)
            {
                // Accumulate 1/dt for velocity
                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    local_dt_dif += _nu / _element_length(cell, point, dim)
                                    / _element_length(cell, point, dim);
                }

                // Take minimum time step
                _local_dt(cell, point)
                    = min(_local_dt(cell, point), 1.0 / local_dt_dif);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_RADLOCALTIMESTEPSIZE_IMPL_HPP
