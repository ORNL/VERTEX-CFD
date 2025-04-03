#ifndef VERTEXCFD_CLOSURE_INDUCTIONCONSTANTSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INDUCTIONCONSTANTSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
InductionConstantSource<EvalType, Traits, NumSpaceDim>::InductionConstantSource(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
{
    const auto ind_input_source
        = closure_params.get<Teuchos::Array<double>>("Induction Source");

    for (int dim = 0; dim < num_space_dim; ++dim)
        _ind_input_source[dim] = ind_input_source[dim];

    // Evaluated fields
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _induction_source, "CONSTANT_SOURCE_induction_");

    this->setName("Induction Constant Source " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void InductionConstantSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
InductionConstantSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _induction_source[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _induction_source[dim](cell, point) = _ind_input_source[dim];
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INDUCTIONCONSTANTSOURCE_IMPL_HPP
