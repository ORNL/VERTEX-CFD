#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleKOmegaDiffusivityCoefficient<EvalType, Traits>::
    IncompressibleKOmegaDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _sigma_k(0.6)
    , _sigma_w(0.5)
    , _diffusivity_var_k("diffusivity_turb_kinetic_energy", ir.dl_scalar)
    , _diffusivity_var_w("diffusivity_turb_specific_dissipation_rate",
                         ir.dl_scalar)
{
    // Check for user-defined coefficients or parameters
    if (user_params.isSublist("Turbulence Parameters"))
    {
        Teuchos::ParameterList turb_list
            = user_params.sublist("Turbulence Parameters");

        if (turb_list.isSublist("K-Omega Parameters"))
        {
            Teuchos::ParameterList k_w_list
                = turb_list.sublist("K-Omega Parameters");

            if (k_w_list.isType<double>("sigma_w"))
            {
                _sigma_w = k_w_list.get<double>("sigma_w");
            }

            if (k_w_list.isType<double>("sigma_k"))
            {
                _sigma_k = k_w_list.get<double>("sigma_k");
            }
        }
    }

    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);

    // Add evaluated fields
    this->addEvaluatedField(_diffusivity_var_k);
    this->addEvaluatedField(_diffusivity_var_w);

    // Closure model name
    this->setName("K-Omega Incompressible Diffusivity Coefficient "
                  + std::to_string(ir.spatial_dimension) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleKOmegaDiffusivityCoefficient<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleKOmegaDiffusivityCoefficient<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _diffusivity_var_k.extent(1);
    const double k_tol = 1e-20;
    const double w_tol = 1e-10;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const scalar_type turb_ratio
                = SmoothMath::max(_turb_kinetic_energy(cell, point), k_tol, 0.0)
                  / SmoothMath::max(
                      _turb_specific_dissipation_rate(cell, point), w_tol, 0.0);
            _diffusivity_var_k(cell, point) = _nu + _sigma_k * turb_ratio;
            _diffusivity_var_w(cell, point) = _nu + _sigma_w * turb_ratio;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_IMPL_HPP
