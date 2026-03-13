#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVARIABLEPROPERTIES_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVARIABLEPROPERTIES_IMPL_HPP

#include "utils/VertexCFD_Utils_PhaseLayout.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Phalanx_DataLayout_DynamicLayout.hpp>

#include <Sacado_Traits.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleLSVOFVariableProperties<EvalType, Traits>::
    IncompressibleLSVOFVariableProperties(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& lsvof_params,
        const std::vector<std::string>& phase_names,
        const bool& build_dxdts,
        const std::string& field_prefix)
    : _rho(field_prefix + "density", ir.dl_scalar)
    , _mu(field_prefix + "dynamic_viscosity", ir.dl_scalar)
    , _dxdt_rho("DXDT_density", ir.dl_scalar)
    , _lsvof_model_type(LSVOFModelType::VOF)
    , _build_dxdts(build_dxdts)
    , _phase_layout(
          Utils::buildPhaseLayout(ir.dl_scalar, phase_names.size() - 1))
    , _alphas(field_prefix + "volume_fractions", _phase_layout)
    , _dxdt_alphas(field_prefix + "DXDT_volume_fractions", _phase_layout)
    , _alpha_n(phase_names[lsvof_params.get<int>("Number of Phases") - 1],
               ir.dl_scalar)
    , _heaviside(field_prefix + "CLS_heaviside", ir.dl_scalar)
    , _phase_rho(Kokkos::ViewAllocateWithoutInitializing("Density"),
                 lsvof_params.get<int>("Number of Phases"))
    , _phase_mu(Kokkos::ViewAllocateWithoutInitializing("Dynamic "
                                                        "viscosity"),
                lsvof_params.get<int>("Number of Phases"))
{
    // Check LSVOF model type
    if (lsvof_params.isType<std::string>("LSVOF Model"))
    {
        const std::string lsvof_model
            = lsvof_params.get<std::string>("LSVOF Model");

        if (lsvof_model == "CLS")
            _lsvof_model_type = LSVOFModelType::CLS;
    }

    // Evaluated fields
    this->addEvaluatedField(_rho);
    this->addEvaluatedField(_mu);
    if (_build_dxdts)
    {
        this->addEvaluatedField(_dxdt_rho);
    }

    // Dependent fields
    if (_lsvof_model_type == LSVOFModelType::VOF)
    {
        this->addDependentField(_alphas);
        if (_build_dxdts)
        {
            this->addDependentField(_dxdt_alphas);
        }
        this->addDependentField(_alpha_n);
    }
    else if (_lsvof_model_type == LSVOFModelType::CLS)
    {
        this->addDependentField(_heaviside);
    }

    auto prho_host
        = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _phase_rho);
    auto pmu_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _phase_mu);

    // read phase properties
    for (int phase = 0; phase < lsvof_params.get<int>("Number of Phases");
         ++phase)
    {
        // Phase numbering starts from 1
        const std::string list_name = "Phase " + std::to_string(phase + 1);
        const Teuchos::ParameterList phase_list
            = lsvof_params.sublist(list_name);
        prho_host[phase] = phase_list.get<double>("Density");
        pmu_host[phase] = phase_list.get<double>("Dynamic viscosity");
    }

    Kokkos::deep_copy(_phase_rho, prho_host);
    Kokkos::deep_copy(_phase_mu, pmu_host);
    this->setName("Incompressible LSVOF Variable Properties");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleLSVOFVariableProperties<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFVariableProperties<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _rho.extent(1);

    // Number of phases. CLS model is hard-coded for two phases for now.
    const size_t nb_phase
        = _lsvof_model_type == LSVOFModelType::VOF ? _alphas.extent(2) : 2;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // calculate mixture properties
            if (_lsvof_model_type == LSVOFModelType::VOF)
            {
                _rho(cell, point) = 0.0;
                _mu(cell, point) = 0.0;
                if (_build_dxdts)
                {
                    _dxdt_rho(cell, point) = 0.0;
                }

                for (size_t phase = 0; phase < nb_phase; ++phase)
                {
                    _rho(cell, point) += _phase_rho[phase]
                                         * _alphas(cell, point, phase);
                    _mu(cell, point) += _phase_mu[phase]
                                        * _alphas(cell, point, phase);
                    if (_build_dxdts)
                    {
                        _dxdt_rho(cell, point)
                            += ((_phase_rho[phase] - _phase_rho[nb_phase])
                                * _dxdt_alphas(cell, point, phase));
                    }
                }
                _rho(cell, point) += _phase_rho[nb_phase]
                                     * _alpha_n(cell, point);
                _mu(cell, point) += _phase_mu[nb_phase] * _alpha_n(cell, point);
            }
            else if (_lsvof_model_type == LSVOFModelType::CLS)
            {
                _rho(cell, point) = _phase_rho[0] * _heaviside(cell, point)
                                    + _phase_rho[1]
                                          * (1.0 - _heaviside(cell, point));
                _mu(cell, point) = _phase_mu[0] * _heaviside(cell, point)
                                   + _phase_mu[1]
                                         * (1.0 - _heaviside(cell, point));
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVARIABLEPROPERTIES_IMPL_HPP
