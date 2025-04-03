#ifndef VERTEXCFD_CLOSURE_VISCOUSHEAT_IMPL_HPP
#define VERTEXCFD_CLOSURE_VISCOUSHEAT_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleViscousHeat<EvalType, Traits, NumSpaceDim>::IncompressibleViscousHeat(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const std::string& gradient_prefix)
    : _viscous_heat_continuity_source("VISCOUS_HEAT_continuity", ir.dl_scalar)
    , _viscous_heat_energy_source("VISCOUS_HEAT_energy", ir.dl_scalar)
    , _rho(fluid_prop.constantDensity())
    , _nu(fluid_prop.constantKinematicViscosity())
{
    // Evaluated fields
    this->addEvaluatedField(_viscous_heat_continuity_source);
    this->addEvaluatedField(_viscous_heat_energy_source);

    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _viscous_heat_momentum_source,
                                   "VISCOUS_HEAT_"
                                   "momentum_");

    // Dependent fields
    Utils::addDependentVectorField(*this,
                                   ir.dl_vector,
                                   _grad_velocity,
                                   gradient_prefix + "GRAD_velocity_");

    this->setName("Incompressible Viscous Heat "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleViscousHeat<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleViscousHeat<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _viscous_heat_continuity_source.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _viscous_heat_continuity_source(cell, point) = 0.0;

            // Reset energy source to zero
            _viscous_heat_energy_source(cell, point) = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    // Calculate deformation tensor component
                    const scalar_type e_ij
                        = 0.5
                          * (_grad_velocity[j](cell, point, i)
                             + _grad_velocity[i](cell, point, j));

                    // Add contribution from each dimension to energy source
                    _viscous_heat_energy_source(cell, point)
                        += 2.0 * _rho * _nu * e_ij * e_ij;
                }

                // No momentum contribution for this source term
                _viscous_heat_momentum_source[i](cell, point) = 0.0;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VISCOUSHEAT_IMPL_HPP
