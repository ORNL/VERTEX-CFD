#ifndef VERTEXCFD_CLOSURE_SCALARTAUSUPG_IMPL_HPP
#define VERTEXCFD_CLOSURE_SCALARTAUSUPG_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
VariableTauSUPG<EvalType, Traits, NumSpaceDim>::VariableTauSUPG(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params,
    const Teuchos::ParameterList& supg_params)
    : _field_name(closure_params.get<std::string>("Field Name"))
    , _tau_supg("tau_supg_" + _field_name, ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
    , _diffusivity("diffusivity_" + _field_name, ir.dl_scalar)
    , _alpha(0.5)
    , _ir_degree(ir.cubature_degree)
{
    // Get tau model type for scalar equation
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<TauModel>(
                Teuchos::tuple<std::string>("Steady", "Transient", "NoSUPG"),
                "Transient"));

        _tau_model = type_validator->getIntegralValue(
            supg_params.get<std::string>("Tau model"));

        // SUPG coefficient
        if (supg_params.isType<double>("SUPG coefficient"))
            _alpha = supg_params.get<double>("SUPG coefficient");
    }

    // Evaluated fields
    this->addEvaluatedField(_tau_supg);

    // Dependent fields
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    this->addDependentField(_element_length);
    this->addDependentField(_diffusivity);

    this->setName("Variable Tau SUPG " + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void VariableTauSUPG<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void VariableTauSUPG<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Get FEM basis order
    _basis_order = workset.bases[_ir_index]->basis_layout->getBasis()->order();

    // Get time step
    const double dt = workset.step_size;
    _time_constant = 4.0 / (dt * dt);

    // Allocate space for thread-local temporaries in parallel for
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg.get_view());
        bytes = scratch_view::shmem_size(
            _tau_supg.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_tau_supg.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
VariableTauSUPG<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _tau_supg.extent(1);
    const double tol = 1.0e-10;
    using Kokkos::sqrt;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Initialize local variables
            auto&& u2_over_h2 = tmp_field(point, U2_OVER_H2);
            u2_over_h2 = 0.0;
            double h2 = 0.0;

            // Compute terms `h^2` and '\frac{u^2}{h^2}`
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                h2 += _element_length(cell, point, dim)
                      * _element_length(cell, point, dim);
                u2_over_h2 += _velocity[dim](cell, point)
                              * _velocity[dim](cell, point)
                              / (_element_length(cell, point, dim)
                                 * _element_length(cell, point, dim));
            }

            // SUPG coefficient for momentum and continuity equations
            if (_tau_model == TauModel::NoSUPG)
            {
                _tau_supg(cell, point) = 0.0;
            }
            else if (_tau_model == TauModel::Steady)
            {
                _tau_supg(cell, point)
                    = 1.0
                      / sqrt(4.0 * u2_over_h2
                             + 9.0 * 16.0 * _diffusivity(cell, point)
                                   * _diffusivity(cell, point) / (h2 * h2)
                             + tol);
            }
            else // Transient
            {
                _tau_supg(cell, point)
                    = 1.0
                      / sqrt(_time_constant + 4.0 * u2_over_h2
                             + 9.0 * 16.0 * _diffusivity(cell, point)
                                   * _diffusivity(cell, point) / (h2 * h2));
            }
            _tau_supg(cell, point) *= _alpha / _basis_order;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SCALARTAUSUPG_IMPL_HPP
