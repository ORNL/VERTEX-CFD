#ifndef VERTEXCFD_CLOSURE_ELECTRICCURRENTDENSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_ELECTRICCURRENTDENSITY_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
ElectricCurrentDensity<EvalType, Traits, NumSpaceDim>::ElectricCurrentDensity(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
    , _sigma(fluid_prop.constantElectricalConductivity())
{
    // Evaluated fields
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _electric_current_density,
                                   "electric_current_density_");

    // Dependent fields
    this->addDependentField(_grad_electric_potential);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _ext_magn_field, "external_magnetic_field_");

    this->setName("Electric Potential Electric Current Density "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ElectricCurrentDensity<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
ElectricCurrentDensity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _electric_current_density[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Electric current density: 2-D case
            // x-component
            _electric_current_density[0](cell, point)
                = -_sigma * _grad_electric_potential(cell, point, 0);
            _electric_current_density[0](cell, point)
                += _sigma * _velocity[1](cell, point)
                   * _ext_magn_field[2](cell, point);
            // y-component
            _electric_current_density[1](cell, point)
                = -_sigma * _grad_electric_potential(cell, point, 1);
            _electric_current_density[1](cell, point)
                += -_sigma * _velocity[0](cell, point)
                   * _ext_magn_field[2](cell, point);

            // Electric current density: 3-D case
            if (num_space_dim == 3)
            {
                // x-component
                _electric_current_density[0](cell, point)
                    -= _sigma * _velocity[2](cell, point)
                       * _ext_magn_field[1](cell, point);
                // y-component
                _electric_current_density[1](cell, point)
                    += _sigma * _velocity[2](cell, point)
                       * _ext_magn_field[0](cell, point);
                // z-component
                _electric_current_density[2](cell, point)
                    = -_sigma * _grad_electric_potential(cell, point, 2);
                _electric_current_density[2](cell, point)
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

#endif // end
       // VERTEXCFD_CLOSURE_ELECTRICCURRENTDENSITY_IMPL_HPP
