#ifndef VERTEXCFD_CLOSURE_VARIABLESUPGFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_VARIABLESUPGFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
VariableSUPGFlux<EvalType, Traits, NumSpaceDim>::VariableSUPGFlux(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _variable_name(closure_params.get<std::string>("Field Name"))
    , _equation_name(closure_params.get<std::string>("Equation Name"))
    , _var_diff_flux("DIFFUSION_FLUX_" + _equation_name, ir.dl_vector)
    , _tau_supg_var("tau_supg_" + _variable_name, ir.dl_scalar)
    , _dxdt_var("DXDT_" + _variable_name, ir.dl_scalar)
    , _var_source("SOURCE_" + _equation_name, ir.dl_scalar)
    , _grad_var("GRAD_" + _variable_name, ir.dl_vector)
{
    // Add contributed fields
    this->addContributedField(_var_diff_flux);

    // Add dependent fields
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    this->addDependentField(_dxdt_var);
    this->addDependentField(_grad_var);
    this->addDependentField(_var_source);
    this->addDependentField(_tau_supg_var);

    this->setName("Variable SUPG Flux " + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void VariableSUPGFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries in parallel for
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg_var.get_view());
        bytes = scratch_view::shmem_size(
            _tau_supg_var.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_tau_supg_var.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
VariableSUPGFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _tau_supg_var.extent(1);

    // Initialize scratch memory
    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg_var.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& var_res = tmp_field(point, VAR_RES);

            // Compute strong residual of scalar equation
            // (non-conservative version)
            var_res = _dxdt_var(cell, point);
            var_res -= _var_source(cell, point);

            for (int i = 0; i < num_space_dim; ++i)
            {
                var_res += _velocity[i](cell, point)
                           * _grad_var(cell, point, i);
            }

            // Loop over cartesian components, build strong residual, and add
            // SUPG contribution to equation
            for (int j = 0; j < num_space_dim; ++j)
            {
                _var_diff_flux(cell, point, j) += _tau_supg_var(cell, point)
                                                  * var_res
                                                  * _velocity[j](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VARIABLESUPGFLUX_IMPL_HPP
