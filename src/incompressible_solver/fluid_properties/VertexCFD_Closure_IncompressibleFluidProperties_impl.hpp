#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEFLUIDPROPERTIES_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEFLUIDPROPERTIES_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace FluidProperties
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleFluidProperties<EvalType, Traits>::IncompressibleFluidProperties(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& fluid_params)
    : _density("density", ir.dl_scalar)
    , _kinematic_viscosity("kinematic_viscosity", ir.dl_scalar)
    , _thermal_conductivity("thermal_conductivity", ir.dl_scalar)
    , _specific_heat_capacity("specific_heat_capacity", ir.dl_scalar)
    , _electrical_conductivity("electrical_conductivity", ir.dl_scalar)
    , _rho(1.0)
    , _nu(std::numeric_limits<double>::quiet_NaN())
    , _k(std::numeric_limits<double>::quiet_NaN())
    , _cp(std::numeric_limits<double>::quiet_NaN())
    , _sigma(std::numeric_limits<double>::quiet_NaN())
    , _fluid_prop_type(FluidPropertyType::constant)
{
    // Evaluated fields
    this->addEvaluatedField(_density);
    this->addEvaluatedField(_kinematic_viscosity);

    // Get fluid property type
    if (fluid_params.isType<std::string>("Fluid Property Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<FluidPropertyType>(
                Teuchos::tuple<std::string>("constant"), "constant"));
        _fluid_prop_type = type_validator->getIntegralValue(
            fluid_params.get<std::string>("Thermal Conductivity Type"));
    }

    // Get values based on the fluid property type
    if (_fluid_prop_type == FluidPropertyType::constant)
    {
        if (fluid_params.isType<double>("Density"))
            _rho = fluid_params.get<double>("Density");
        _nu = fluid_params.get<double>("Kinematic viscosity");
        if (fluid_params.isType<double>("Thermal conductivity"))
        {
            _k = fluid_params.get<double>("Thermal conductivity");
            this->addEvaluatedField(_thermal_conductivity);
        }
        if (fluid_params.isType<double>("Specific heat capacity"))
        {
            _cp = fluid_params.get<double>("Specific heat capacity");
            this->addEvaluatedField(_specific_heat_capacity);
        }
        if (fluid_params.isType<double>("Electrical conductivity"))
        {
            _sigma = fluid_params.get<double>("Electrical conductivity");
            this->addEvaluatedField(_electrical_conductivity);
        }
    }

    this->setName("Fluid properties");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleFluidProperties<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleFluidProperties<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _kinematic_viscosity.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            if (_fluid_prop_type == FluidPropertyType::constant)
            {
                _density(cell, point) = _rho;
                _kinematic_viscosity(cell, point) = _nu;
                if (!std::isnan(_k))
                    _thermal_conductivity(cell, point) = _k;
                if (!std::isnan(_cp))
                    _specific_heat_capacity(cell, point) = _cp;
                if (!std::isnan(_sigma))
                    _electrical_conductivity(cell, point) = _sigma;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace FluidProperties
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEFLUIDPROPERTIES_IMPL_HPP
