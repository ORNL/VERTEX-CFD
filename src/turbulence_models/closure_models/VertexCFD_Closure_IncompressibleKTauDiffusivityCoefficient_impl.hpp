#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleKTauDiffusivityCoefficient<EvalType, Traits>::
    IncompressibleKTauDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& user_params)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _nu("kinematic_viscosity", ir.dl_scalar)
    , _sigma_k(0.6)
    , _sigma_t(0.5)
    , _diffusivity_var_k("diffusivity_turb_kinetic_energy", ir.dl_scalar)
    , _diffusivity_var_t("diffusivity_turb_specific_dissipation_rate",
                         ir.dl_scalar)
{
    // Check for user-defined coefficients or parameters
    if (user_params.isSublist("Turbulence Parameters"))
    {
        Teuchos::ParameterList turb_list
            = user_params.sublist("Turbulence Parameters");

        if (turb_list.isSublist("K-Tau Parameters"))
        {
            Teuchos::ParameterList k_t_list
                = turb_list.sublist("K-Tau Parameters");

            if (k_t_list.isType<double>("sigma_t"))
            {
                _sigma_t = k_t_list.get<double>("sigma_t");
            }

            if (k_t_list.isType<double>("sigma_k"))
            {
                _sigma_k = k_t_list.get<double>("sigma_k");
            }
        }
    }

    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);
    this->addDependentField(_nu);

    // Add evaluated fields
    this->addEvaluatedField(_diffusivity_var_k);
    this->addEvaluatedField(_diffusivity_var_t);

    // Closure model name
    this->setName("K-Tau Incompressible Diffusivity Coefficient "
                  + std::to_string(ir.spatial_dimension) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleKTauDiffusivityCoefficient<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleKTauDiffusivityCoefficient<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _diffusivity_var_k.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _diffusivity_var_k(cell, point)
                = _nu(cell, point)
                  + _sigma_k * _turb_kinetic_energy(cell, point)
                        * _turb_specific_dissipation_rate(cell, point);
            _diffusivity_var_t(cell, point)
                = _nu(cell, point)
                  + _sigma_t * _turb_kinetic_energy(cell, point)
                        * _turb_specific_dissipation_rate(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_IMPL_HPP
