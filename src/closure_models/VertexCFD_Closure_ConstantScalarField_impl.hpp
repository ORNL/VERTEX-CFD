#ifndef VERTEXCFD_CLOSURE_CONSTANTSCALARFIELD_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONSTANTSCALARFIELD_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ConstantScalarField<EvalType, Traits>::ConstantScalarField(
    const panzer::IntegrationRule& ir,
    const std::string& field_name,
    const double field_value)
    : _scalar_field(field_name, ir.dl_scalar)
    , _field_value(field_value)
{
    this->addEvaluatedField(_scalar_field);

    this->setName("Constant Scalar Field \"" + field_name + "\"");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ConstantScalarField<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void ConstantScalarField<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _scalar_field.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point),
        [&](const int point) { _scalar_field(cell, point) = _field_value; });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONSTANTSCALARFIELD_IMPL_HPP
