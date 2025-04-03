#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLEDIFFUSIONFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLEDIFFUSIONFLUX_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleVariableDiffusionFlux<EvalType, Traits, NumSpaceDim>::
    IncompressibleVariableDiffusionFlux(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& flux_prefix,
        const std::string& gradient_prefix)
    : _variable_name(closure_params.get<std::string>("Field Name"))
    , _equation_name(closure_params.get<std::string>("Equation Name"))
    , _var_diff_flux(flux_prefix + "DIFFUSION_FLUX_" + _equation_name,
                     ir.dl_vector)
    , _diffusivity_var("diffusivity_" + _variable_name, ir.dl_scalar)
    , _grad_var(gradient_prefix + "GRAD_" + _variable_name, ir.dl_vector)
{
    // Add evaludated fields
    this->addEvaluatedField(_var_diff_flux);

    // Add dependent fields
    this->addDependentField(_diffusivity_var);
    this->addDependentField(_grad_var);

    // Closure model name
    this->setName(_equation_name + " Incompressible Diffusion Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleVariableDiffusionFlux<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleVariableDiffusionFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _var_diff_flux.extent(1);
    const int num_grad_dim = _var_diff_flux.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Loop over spatial dimension
            for (int i = 0; i < num_grad_dim; ++i)
            {
                _var_diff_flux(cell, point, i) = _diffusivity_var(cell, point)
                                                 * _grad_var(cell, point, i);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLEDIFFUSIONFLUX_IMPL_HPP
