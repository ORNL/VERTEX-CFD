#ifndef VERTEXCFD_CLOSURE_METRICTENSORELEMENTLENGTH_IMPL_HPP
#define VERTEXCFD_CLOSURE_METRICTENSORELEMENTLENGTH_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
MetricTensorElementLength<EvalType, Traits>::MetricTensorElementLength(
    const panzer::IntegrationRule& ir, const std::string& prefix)
    : _element_length(prefix + "element_length", ir.dl_vector)
    , _metric_tensor("metric_tensor", ir.dl_tensor)
{
    this->addEvaluatedField(_element_length);
    this->addDependentField(_metric_tensor);
    this->setName("Metric Tensor Element Length");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MetricTensorElementLength<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
MetricTensorElementLength<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _metric_tensor.extent(1);
    const int num_space_dim = _metric_tensor.extent(2);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             using std::sqrt;
                             for (int i = 0; i < num_space_dim; ++i)
                             {
                                 _element_length(cell, point, i)
                                     = sqrt(_metric_tensor(cell, point, i, i));
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_METRICTENSORELEMENTLENGTH_IMPL_HPP
