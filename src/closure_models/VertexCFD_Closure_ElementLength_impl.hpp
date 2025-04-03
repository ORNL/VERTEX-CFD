#ifndef VERTEXCFD_CLOSURE_ELEMENTLENGTH_IMPL_HPP
#define VERTEXCFD_CLOSURE_ELEMENTLENGTH_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ElementLength<EvalType, Traits>::ElementLength(const panzer::IntegrationRule& ir,
                                               const std::string& prefix)
    : _element_length(prefix + "element_length", ir.dl_vector)
{
    this->addEvaluatedField(_element_length);
    this->setName("Element Length");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ElementLength<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    // TODO don't assume a 0 basis index.
    _grad_basis = this->wda(workset).bases[0]->grad_basis;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void ElementLength<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _element_length.extent(1);
    const int num_space_dim = _element_length.extent(2);
    const int num_basis = _grad_basis.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::pow;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                double grad_basis_squared = 0.0;
                for (int basis = 0; basis < num_basis; ++basis)
                {
                    grad_basis_squared
                        += pow(_grad_basis(cell, basis, point, dim), 2);
                }
                _element_length(cell, point, dim)
                    = pow(grad_basis_squared, -0.5);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_ELEMENTLENGTH_IMPL_HPP
