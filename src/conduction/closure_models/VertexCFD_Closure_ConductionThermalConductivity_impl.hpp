#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ConductionThermalConductivity<EvalType, Traits>::ConductionThermalConductivity(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _thermal_conductivity("thermal_conductivity", ir.dl_scalar)
    , _temperature("temperature", ir.dl_scalar)
    , _num_space_dim(ir.spatial_dimension)
    , _k0(std::numeric_limits<double>::quiet_NaN())
    , _therm_cond_type(ThermCondType::constant)
{
    // Evaluated fields
    this->addEvaluatedField(_thermal_conductivity);

    // Get thermal conductivity type
    if (closure_params.isType<std::string>("Thermal Conductivity Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<ThermCondType>(
                Teuchos::tuple<std::string>("constant", "inverse_proportional"),
                "constant"));
        _therm_cond_type = type_validator->getIntegralValue(
            closure_params.get<std::string>("Thermal Conductivity Type"));
    }

    // Get values based on the thermal conductivity type
    if (_therm_cond_type == ThermCondType::constant)
    {
        _k0 = closure_params.get<double>("Thermal Conductivity Value");
    }
    else
    {
        _k0 = closure_params.get<double>("Thermal Conductivity Coefficient");
    }

    // Dependent fields
    this->addDependentField(_temperature);

    this->setName("Conduction Thermal Conductivity "
                  + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConductionThermalConductivity<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ConductionThermalConductivity<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _thermal_conductivity.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             if (_therm_cond_type == ThermCondType::constant)
                             {
                                 _thermal_conductivity(cell, point) = _k0;
                             }
                             else
                             {
                                 _thermal_conductivity(cell, point)
                                     = _k0 / _temperature(cell, point);
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONTHERMALCONDUCTIVITY_IMPL_HPP
