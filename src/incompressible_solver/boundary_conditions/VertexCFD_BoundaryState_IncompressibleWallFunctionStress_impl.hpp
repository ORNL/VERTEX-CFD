#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEWALLFUNCTIONSTRESS_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEWALLFUNCTIONSTRESS_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleWallFunctionStress<EvalType, Traits, NumSpaceDim>::
    IncompressibleWallFunctionStress(const panzer::IntegrationRule& ir,
                                     const std::string& flux_prefix)
    : _boundary_u_tau("BOUNDARY_friction_velocity", ir.dl_scalar)
    , _boundary_y_plus("BOUNDARY_y_plus", ir.dl_scalar)
{
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _wall_func_shear_stress,
                                   flux_prefix + "momentum_");

    // Turbulence quantites obtained from corresponding wall function boundary
    // state used for turbulence quantities.
    this->addDependentField(_boundary_u_tau);
    this->addDependentField(_boundary_y_plus);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    this->setName("Boundary State Incompressible Wall Function Stress "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleWallFunctionStress<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleWallFunctionStress<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_y_plus.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             // Set wall function shear stress components
                             for (int dim = 0; dim < num_space_dim; ++dim)
                             {
                                 _wall_func_shear_stress[dim](cell, point)
                                     = _boundary_u_tau(cell, point)
                                       / _boundary_y_plus(cell, point)
                                       * _velocity[dim](cell, point);
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEWALLFUNCTIONSTRESS_IMPL_HPP
