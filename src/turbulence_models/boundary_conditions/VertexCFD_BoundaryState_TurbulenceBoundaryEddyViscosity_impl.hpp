#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
TurbulenceBoundaryEddyViscosity<EvalType, Traits>::TurbulenceBoundaryEddyViscosity(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& bc_params,
    const std::string& flux_prefix)
    : _boundary_nu_t(flux_prefix + "turbulent_eddy_viscosity", ir.dl_scalar)
    , _interior_nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _wall_func_nu_t("wall_func_turbulent_eddy_viscosity", ir.dl_scalar)
    , _wall_func(false)
{
    // Only check boundary if parameter list is populated
    // (it should be empty in some models, i.e. WALE)
    if (bc_params.numParams() > 0)
    {
        // Check boundary condition type is a wall function
        const std::string bc_type = bc_params.get<std::string>("Type");

        if (std::string::npos != bc_type.find("Wall Function"))
        {
            _wall_func = true;
        }
    }

    // Add evaluated fields
    this->addEvaluatedField(_boundary_nu_t);

    // Add dependent fields
    if (_wall_func)
    {
        this->addDependentField(_wall_func_nu_t);
    }
    else
    {
        this->addDependentField(_interior_nu_t);
    }

    this->setName("Boundary State Turbulence Eddy Viscosity");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void TurbulenceBoundaryEddyViscosity<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
TurbulenceBoundaryEddyViscosity<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_nu_t.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set boundary nu_t according to BC type
            if (_wall_func)
            {
                _boundary_nu_t(cell, point) = _wall_func_nu_t(cell, point);
            }
            else
            {
                _boundary_nu_t(cell, point) = _interior_nu_t(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_IMPL_HPP
