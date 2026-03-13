#ifndef VERTEXCFD_CLOSURE_RADTRANSMUTATIONSOURCEEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_RADTRANSMUTATIONSOURCEEXACTSOLUTION_IMPL_HPP

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
RADTransmutationSourceExactSolution<EvalType, Traits>::
    RADTransmutationSourceExactSolution(
        const panzer::IntegrationRule& ir,
        const SpeciesProperties::ConstantSpeciesProperties& species_prop,
        const Teuchos::ParameterList& closure_params)
    : _num_species(species_prop.numSpecies())
    , _flux("neutron_flux", ir.dl_scalar)
    , _mic_cross_section(Kokkos::ViewAllocateWithoutInitializing("microscopic_"
                                                                 "cross_"
                                                                 "section"),
                         _num_species,
                         _num_species)
    , _exp_A_p(Kokkos::ViewAllocateWithoutInitializing("exp_A_p"),
               _num_species,
               _num_species)
{
    // Copy number of atoms vector from host to device
    auto mic_cross_sec_host
        = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _mic_cross_section);
    mic_cross_sec_host = species_prop.microscopicCrossSection();
    Kokkos::deep_copy(_mic_cross_section, mic_cross_sec_host);
    Kokkos::deep_copy(_exp_A_p, mic_cross_sec_host);

    _a = closure_params.get<double>("Time Shape Function Ramp Scale");
    _b = closure_params.get<double>("Time Shape Function Ramp Offset");
    _kappa = closure_params.get<double>("Time Shape Function Decline Scale");
    _beta = closure_params.get<double>("Time Shape Function Decline Offset");
    _flux_amp = closure_params.get<double>("Neutron Flux Amplitude");
    _calc_flux = closure_params.get<bool>("Calculate Analytical Neutron Flux");
    _ts_order = closure_params.get<int>("Taylor Series Expansion Order");

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

    this->setName("RAD Transmutation Source Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void RADTransmutationSourceExactSolution<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _time = workset.time;

    _c1 = log(cosh(_a * _time - _b)) - log(cosh(_kappa * _time - _beta));
    _c2 = log(cosh(-_b)) - log(cosh(-_beta));
    _c3 = _flux_amp / 4.0 * (_c1 - _c2);

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
RADTransmutationSourceExactSolution<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _exact_species[0].extent(1);

    using Kokkos::cosh;
    using Kokkos::log;
    using Kokkos::tanh;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate _mic_cross_section * Phi
            for (int i = 0; i < _num_species; ++i)
            {
                for (int j = 0; j < _num_species; ++j)
                {
                    if constexpr (std::is_same_v<scalar_type, double>)
                        _exp_A_p(i, j) = _mic_cross_section(i, j) * _c3;
                    else
                        _exp_A_p(i, j) = (_mic_cross_section(i, j) * _c3).val();
                }
            }

            // Define temporary arrays for matrix exponential
            Kokkos::Array<double, VertexCFD::Constants::MAX_NUM_VIEW> term;
            Kokkos::Array<double, VertexCFD::Constants::MAX_NUM_VIEW> tmp;

            for (int i = 0; i < _num_species; ++i)
            {
                _exact_species[i](cell, point) = _initial_amp[i];
                term[i] = _initial_amp[i];
                tmp[i] = 0.0;
            }

            // Taylor series expansion of matrix exponential
            for (int k = 1; k <= _ts_order; ++k)
            {
                for (int i = 0; i < _num_species; ++i)
                {
                    tmp[i] = 0.0;
                    for (int j = 0; j < _num_species; ++j)
                        tmp[i] += _exp_A_p(i, j) * term[j];
                }

                for (int i = 0; i < _num_species; ++i)
                {
                    term[i] = tmp[i] / static_cast<double>(k);
                    _exact_species[i](cell, point) += term[i];
                }
            }

            if (_calc_flux)
                _flux(cell, point) = _flux_amp / 2.0
                                     * ((tanh(_a * _time - _b) + 1.0)
                                        - (tanh(_kappa * _time - _beta) + 1.0));
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_RADTRANSMUTATIONSOURCEEXACTSOLUTION_IMPL_HPP