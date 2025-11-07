#ifndef VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ConductionVolumetricSource<EvalType, Traits>::ConductionVolumetricSource(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _source("SOURCE_energy", ir.dl_scalar)
    , _num_space_dim(ir.spatial_dimension)
    , _ir_degree(ir.cubature_degree)
    , _q(closure_params.get<double>("Volumetric Heat Source Value"))
    , _x_left(std::numeric_limits<double>::signaling_NaN())
    , _x_right(std::numeric_limits<double>::signaling_NaN())
    , _heat_source_type(HeatSourceType::constant)
{
    // Evaluated fields
    this->addEvaluatedField(_source);

    // Get heat source type
    if (closure_params.isType<std::string>("Heat Source Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<HeatSourceType>(
                Teuchos::tuple<std::string>("constant", "xlinear"),
                "constant"));
        _heat_source_type = type_validator->getIntegralValue(
            closure_params.get<std::string>("Heat Source Type"));
    }

    // Get parameters for each source type
    if (_heat_source_type == HeatSourceType::xlinear)
    {
        _x_left = closure_params.get<double>("X-value of the left boundary");
        _x_right = closure_params.get<double>("X-value of the right boundary");
    }

    // Closure model name
    this->setName("Conduction Volumetric Heat Source "
                  + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConductionVolumetricSource<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConductionVolumetricSource<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ConductionVolumetricSource<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _source.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             if (_heat_source_type == HeatSourceType::constant)
                             {
                                 _source(cell, point) = _q;
                             }
                             else
                             {
                                 const double x = _ip_coords(cell, point, 0);
                                 _source(cell, point) = _q * (x - _x_left)
                                                        / (_x_right - _x_left);
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONVOLUMETRICSOURCE_IMPL_HPP
