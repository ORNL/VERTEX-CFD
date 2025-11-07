#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ConductionTimeStepSize<EvalType, Traits>::ConductionTimeStepSize(
    const panzer::IntegrationRule& ir)
    : _local_dt("local_dt", ir.dl_scalar)
    , _thermal_conductivity("thermal_conductivity", ir.dl_scalar)
    , _specific_heat("solid_specific_heat_capacity", ir.dl_scalar)
    , _solid_density("solid_density", ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
{
    // Add evaluated field
    this->addEvaluatedField(_local_dt);

    // Add dependent fields
    this->addDependentField(_element_length);
    this->addDependentField(_solid_density);
    this->addDependentField(_thermal_conductivity);
    this->addDependentField(_specific_heat);

    this->setName("Conduction Time-Step Size");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConductionTimeStepSize<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ConductionTimeStepSize<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _local_dt.extent(1);
    const int num_space_dim = _element_length.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _local_dt(cell, point) = 0.0;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _local_dt(cell, point)
                    += _element_length(cell, point, dim)
                       * _element_length(cell, point, dim)
                       * _solid_density(cell, point)
                       * _specific_heat(cell, point)
                       / (2.0 * _thermal_conductivity(cell, point));
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONDUCTIONTIMESTEPSIZE_IMPL_HPP
