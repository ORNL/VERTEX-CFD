#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleKEpsilonDiffusivityCoefficient<EvalType, Traits>::
    IncompressibleKEpsilonDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const double sigma_k,
        const double sigma_e,
        const std::string field_prefix)
    : _nu_t(field_prefix + "turbulent_eddy_viscosity", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _sigma_k(sigma_k)
    , _sigma_e(sigma_e)
    , _num_grad_dim(ir.spatial_dimension)
    , _diffusivity_var_k("diffusivity_turb_kinetic_energy", ir.dl_scalar)
    , _diffusivity_var_e("diffusivity_turb_dissipation_rate", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_nu_t);

    // Add evaluated fields
    this->addEvaluatedField(_diffusivity_var_k);
    this->addEvaluatedField(_diffusivity_var_e);

    // Closure model name
    this->setName("K-Epsilon Incompressible Diffusivity Coefficient "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleKEpsilonDiffusivityCoefficient<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleKEpsilonDiffusivityCoefficient<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _diffusivity_var_k.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             _diffusivity_var_k(cell, point)
                                 = (_nu + _nu_t(cell, point) / _sigma_k);
                             _diffusivity_var_e(cell, point)
                                 = (_nu + _nu_t(cell, point) / _sigma_e);
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_IMPL_HPP
