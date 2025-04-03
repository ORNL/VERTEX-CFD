#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleSSTDiffusivityCoefficient<EvalType, Traits>::
    IncompressibleSSTDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params)
    : _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _sst_blending_function("sst_blending_function", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _sigma_k1(0.85)
    , _sigma_k2(1.0)
    , _sigma_w1(0.5)
    , _sigma_w2(0.856)
    , _diffusivity_var_k("diffusivity_turb_kinetic_energy", ir.dl_scalar)
    , _diffusivity_var_w("diffusivity_turb_specific_dissipation_rate",
                         ir.dl_scalar)
{
    // Check for user-defined coefficients or parameters
    if (user_params.isSublist("Turbulence Parameters"))
    {
        Teuchos::ParameterList turb_list
            = user_params.sublist("Turbulence Parameters");

        if (turb_list.isSublist("SST Parameters"))
        {
            Teuchos::ParameterList sst_list
                = turb_list.sublist("SST Parameters");

            if (sst_list.isType<double>("sigma_k1"))
            {
                _sigma_k1 = sst_list.get<double>("sigma_k1");
            }

            if (sst_list.isType<double>("sigma_k2"))
            {
                _sigma_k2 = sst_list.get<double>("sigma_k2");
            }

            if (sst_list.isType<double>("sigma_w1"))
            {
                _sigma_w1 = sst_list.get<double>("sigma_w1");
            }

            if (sst_list.isType<double>("sigma_w2"))
            {
                _sigma_w2 = sst_list.get<double>("sigma_w2");
            }
        }
    }

    // Add dependent fields
    this->addDependentField(_nu_t);
    this->addDependentField(_sst_blending_function);

    // Add evaluated fields
    this->addEvaluatedField(_diffusivity_var_k);
    this->addEvaluatedField(_diffusivity_var_w);

    // Closure model name
    this->setName("Incompressible SST Diffusivity Coefficient "
                  + std::to_string(ir.spatial_dimension) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleSSTDiffusivityCoefficient<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleSSTDiffusivityCoefficient<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _diffusivity_var_k.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _diffusivity_var_k(cell, point)
                = _nu
                  + (_sigma_k1 * _sst_blending_function(cell, point)
                     + _sigma_k2 * (1.0 - _sst_blending_function(cell, point)))
                        * _nu_t(cell, point);
            _diffusivity_var_w(cell, point)
                = _nu
                  + (_sigma_w1 * _sst_blending_function(cell, point)
                     + _sigma_w2 * (1.0 - _sst_blending_function(cell, point)))
                        * _nu_t(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_IMPL_HPP
