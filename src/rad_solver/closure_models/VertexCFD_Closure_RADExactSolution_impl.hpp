#ifndef VERTEXCFD_CLOSURE_RADEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADEXACTSOLUTION_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
RADExactSolution<EvalType, Traits, NumSpaceDim>::RADExactSolution(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const Teuchos::ParameterList& closure_params)
    : _num_species(species_prop.numSpecies())
    , _ir_degree(ir.cubature_degree)
    , _nu(species_prop.constantDiffusionCoefficient())
    , _bateman_matrix(Kokkos::ViewAllocateWithoutInitializing("bateman_"
                                                              "matrix"),
                      _num_species,
                      _num_species)
    , _build_reaction(species_prop.buildReaction())
    , _build_advection(species_prop.buildAdvection())
    , _build_diffusion(species_prop.buildDiffusion())
{
    // Copy bateman matrix from host to device
    auto bateman_matrix_host
        = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _bateman_matrix);
    bateman_matrix_host = species_prop.batemanMatrix();
    Kokkos::deep_copy(_bateman_matrix, bateman_matrix_host);

    const auto scale
        = closure_params.get<Teuchos::Array<double>>("Initial Amplitude");
    for (int i = 0; i < _num_species; ++i)
    {
        _scale[i] = scale[i];
    }

    if (_build_advection || _build_diffusion)
    {
        _init_loc = closure_params.get<double>("Initial Location");
        _sigma = closure_params.get<double>("Sigma");
    }

    Utils::addEvaluatedScalarFieldView(
        *this, ir.dl_scalar, _num_species, _exact_species, "Exact_species_");

    if (_build_advection)
        Utils::addDependentVectorField(
            *this, ir.dl_scalar, _velocity, "velocity_");

    this->setName("RAD Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void RADExactSolution<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void RADExactSolution<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size
            = Kokkos::dimension_scalar(_exact_species[0].get_view());
        bytes = scratch_view::shmem_size(
            _exact_species[0].extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_exact_species[0].extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    _time = workset.time;
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
RADExactSolution<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _exact_species[0].extent(1);

    using Kokkos::exp;
    using Kokkos::pow;
    using Kokkos::sqrt;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size
            = Kokkos::dimension_scalar(_exact_species[0].get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& gaussian = tmp_field(point, GAUSSIAN);
            auto&& denum = tmp_field(point, DENUM);

            auto&& sum_k = tmp_field(point, SUM_K);
            auto&& prod_l = tmp_field(point, PROD_L);
            auto&& lmbd_prod = tmp_field(point, LMBD_PROD);

            gaussian = 0.0;
            denum = 0.0;

            sum_k = 0.0;
            prod_l = 1.0;
            lmbd_prod = 1.0;
            if (_build_advection || _build_diffusion)
            {
                for (int num = 0; num < _num_species; ++num)
                {
                    _exact_species[num](cell, point) = 0.0;
                    for (int dim = 0; dim < 1; ++dim)
                    {
                        gaussian = _ip_coords(cell, point, dim) - _init_loc;

                        if (_build_advection)
                            gaussian -= _time * _velocity[dim](cell, point);

                        denum = _sigma * _sigma;

                        if (_build_diffusion)
                            denum += 2 * _nu * _time;

                        for (int i = 0; i < 1000; ++i)
                        {
                            _exact_species[num](cell, point)
                                += exp(-pow(gaussian + i, 2) * 0.5 / (denum));
                        }

                        if (_build_diffusion)
                            _exact_species[num](cell, point)
                                *= (1.0
                                    / sqrt(1.0
                                           * (1.0
                                              + 2.0 * _nu * _time
                                                    / (_sigma * _sigma))));
                    }

                    if (!_build_reaction)
                        _exact_species[num](cell, point) *= _scale[num];
                }
            }

            if (_build_reaction)
            {
                for (int num = 1; num < _num_species + 1; ++num)
                {
                    lmbd_prod = 1.0;
                    for (int i = 0; i < num - 1; i++)
                    {
                        lmbd_prod *= -_bateman_matrix(i, i);
                    }
                    sum_k = 0;
                    for (int i = 0; i < num; i++)
                    {
                        prod_l = 1.0;
                        for (int j = 0; j < num; j++)
                        {
                            if (j != i)
                                prod_l *= (-_bateman_matrix(j, j)
                                           + _bateman_matrix(i, i));
                        }
                        sum_k += exp(+_bateman_matrix(i, i) * _time) / prod_l;
                    }
                    if (_build_advection || _build_diffusion)
                    {
                        _exact_species[num - 1](cell, point)
                            *= _scale[0] * lmbd_prod * sum_k;
                    }
                    else
                    {
                        _exact_species[num - 1](cell, point)
                            = _scale[0] * lmbd_prod * sum_k;
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_RADEXACTSOLUTION_IMPL_HPP