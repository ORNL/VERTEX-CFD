#ifndef VERTEXCFD_CLOSURE_RADFISSIONSOURCEEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADFISSIONSOURCEEXACTSOLUTION_IMPL_HPP

#include "utils/VertexCFD_Utils_ScalarFieldView.hpp"

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
template<class EvalType, class Traits>
RADFissionSourceExactSolution<EvalType, Traits>::RADFissionSourceExactSolution(
    const panzer::IntegrationRule& ir,
    const SpeciesProperties::ConstantSpeciesProperties& species_prop,
    const Teuchos::ParameterList& closure_params)
    : _num_species(species_prop.numSpecies())
    , _flux("neutron_flux", ir.dl_scalar)
    , _xs(species_prop.fissionCrossSection())
    , _avagadro(6.02214076e23)
    , _gamma(Kokkos::ViewAllocateWithoutInitializing("atoms_per_species"),
             _num_species)
{
    // Copy number of atoms vector from host to device
    auto gamma_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _gamma);
    gamma_host = species_prop.atomsPerSpecies();
    Kokkos::deep_copy(_gamma, gamma_host);

    _a = closure_params.get<double>("Time Shape Function Ramp Scale");
    _b = closure_params.get<double>("Time Shape Function Ramp Offset");
    _kappa = closure_params.get<double>("Time Shape Function Decline Scale");
    _beta = closure_params.get<double>("Time Shape Function Decline Offset");
    _flux_amp = closure_params.get<double>("Neutron Flux Amplitude");
    _calc_flux = closure_params.get<bool>("Calculate Analytical Neutron Flux");
    const auto initial_amp = closure_params.get<Teuchos::Array<double>>(
        "Initial Species Concentration");
    for (int i = 0; i < _num_species; ++i)
    {
        _initial_amp[i] = initial_amp[i];
    }

    Utils::addEvaluatedScalarFieldView(
        *this, ir.dl_scalar, _num_species, _exact_species, "Exact_species_");

    if (_calc_flux)
        this->addEvaluatedField(_flux);

    this->setName("RAD Fission Source Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADFissionSourceExactSolution<EvalType, Traits>::evaluateFields(
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
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
RADFissionSourceExactSolution<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _exact_species[0].extent(1);

    using Kokkos::cosh;
    using Kokkos::log;
    using Kokkos::tanh;

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
            auto&& const_s = tmp_field(point, CONSTS);
            auto&& const_c = tmp_field(point, CONSTC);
            if (_calc_flux)
                _flux(cell, point) = _flux_amp / 2.0
                                     * ((tanh(_a * _time - _b) + 1.0)
                                        - (tanh(_kappa * _time - _beta) + 1.0));
            for (int num = 0; num < _num_species; ++num)
            {
                const_s = _flux_amp * _gamma[num] * _xs / _avagadro;
                const_c = -const_s * log(cosh(_b)) / (2 * _a)
                          + const_s * log(cosh(-_beta)) / (2 * _kappa)
                          - _initial_amp[num];
                _exact_species[num](cell, point)
                    = const_s * log(cosh(_b - _a * _time)) / (2 * _a)
                      - const_s * log(cosh(_kappa * _time - _beta))
                            / (2 * _kappa)
                      + const_c;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_RADFISSIONSOURCEEXACTSOLUTION_IMPL_HPP