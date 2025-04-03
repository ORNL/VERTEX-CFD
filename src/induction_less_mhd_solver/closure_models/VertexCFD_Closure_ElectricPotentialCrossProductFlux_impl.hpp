#ifndef VERTEXCFD_CLOSURE_ELECTRICPOTENTIALCROSSPRODUCTFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_ELECTRICPOTENTIALCROSSPRODUCTFLUX_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
ElectricPotentialCrossProductFlux<EvalType, Traits, NumSpaceDim>::
    ElectricPotentialCrossProductFlux(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const std::string& flux_prefix,
        const std::string& field_prefix)
    : _electric_potential_flux(
          flux_prefix + "ELECTRIC_POTENTIAL_FLUX_electric_potential_equation",
          ir.dl_vector)
    , _sigma(fluid_prop.constantElectricalConductivity())
{
    // Evaluated fields
    this->addContributedField(_electric_potential_flux);

    // Dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _ext_magn_field, "external_magnetic_field_");

    this->setName("Electric Potential Cross Product Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ElectricPotentialCrossProductFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
ElectricPotentialCrossProductFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _electric_potential_flux.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // 2-D case (x- and y-components only)
            _electric_potential_flux(cell, point, 0)
                += _sigma * _velocity[1](cell, point)
                   * _ext_magn_field[2](cell, point);
            _electric_potential_flux(cell, point, 1)
                -= _sigma * _velocity[0](cell, point)
                   * _ext_magn_field[2](cell, point);

            // 3-D case
            if (num_space_dim == 3)
            {
                // x-component
                _electric_potential_flux(cell, point, 0)
                    -= _sigma * _velocity[2](cell, point)
                       * _ext_magn_field[1](cell, point);
                // y-component
                _electric_potential_flux(cell, point, 1)
                    += _sigma * _velocity[2](cell, point)
                       * _ext_magn_field[0](cell, point);
                // z-component
                _electric_potential_flux(cell, point, 2)
                    += _sigma
                       * (_velocity[0](cell, point)
                              * _ext_magn_field[1](cell, point)
                          - _velocity[1](cell, point)
                                * _ext_magn_field[0](cell, point));
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_ELECTRICPOTENTIALCROSSPRODUCTFLUX_IMPL_HPP
