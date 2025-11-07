#ifndef VERTEXCFD_CLOSURE_SOLIDELECTRICCONDUCTIVITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_SOLIDELECTRICCONDUCTIVITY_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
SolidElectricConductivity<EvalType, Traits>::SolidElectricConductivity(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _sigma("solid_electric_conductivity", ir.dl_scalar)
    , _electric_potential("electric_potential", ir.dl_scalar)
    , _sigma0(std::numeric_limits<double>::quiet_NaN())
    , _elec_cond_type(ElecCondType::constant)
{
    // Evaluated field
    this->addEvaluatedField(_sigma);

    // Get electric conductivity type
    if (closure_params.isType<std::string>("Electric Conductivity Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<ElecCondType>(
                Teuchos::tuple<std::string>("constant", "inverse_proportional"),
                "constant"));
        _elec_cond_type = type_validator->getIntegralValue(
            closure_params.get<std::string>("Electric Conductivity Type"));
    }

    // Get values based on the electric conductivity type
    if (_elec_cond_type == ElecCondType::constant)
    {
        _sigma0 = closure_params.get<double>("Electric Conductivity Value");
    }
    else if (_elec_cond_type == ElecCondType::inverse_proportional)
    {
        _sigma0
            = closure_params.get<double>("Electric Conductivity Coefficient");
        // Dependent field
        this->addDependentField(_electric_potential);
    }

    this->setName("Solid Electric Conductivity");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void SolidElectricConductivity<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    if (_elec_cond_type == ElecCondType::constant)
        _sigma.deep_copy(_sigma0);
    else if (_elec_cond_type == ElecCondType::inverse_proportional)
    {
        auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
            workset.num_cells);
        Kokkos::parallel_for(this->getName(), policy, *this);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
SolidElectricConductivity<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _sigma.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _sigma(cell, point) = _sigma0 / _electric_potential(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SOLIDELECTRICCONDUCTIVITY_IMPL_HPP
