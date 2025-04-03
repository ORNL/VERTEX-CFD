#ifndef VERTEXCFD_CLOSURE_DIVERGENCECLEANINGSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_DIVERGENCECLEANINGSOURCE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
DivergenceCleaningSource<EvalType, Traits, NumSpaceDim>::DivergenceCleaningSource(
    const panzer::IntegrationRule& ir)
    : _div_cleaning_potential_source(
          "DIV_CLEANING_SOURCE_magnetic_correction_potential", ir.dl_scalar)
    , _grad_scalar_magnetic_potential("GRAD_scalar_magnetic_potential",
                                      ir.dl_vector)
{
    // Evaluated fields
    this->addEvaluatedField(_div_cleaning_potential_source);

    // Dependent fields
    this->addDependentField(_grad_scalar_magnetic_potential);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    this->setName("Divergence Cleaning Source " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void DivergenceCleaningSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
DivergenceCleaningSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_scalar_magnetic_potential.extent(1);
    const int num_grad_dim = _grad_scalar_magnetic_potential.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _div_cleaning_potential_source(cell, point) = 0.0;

            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                _div_cleaning_potential_source(cell, point)
                    -= _velocity[dim](cell, point)
                       * _grad_scalar_magnetic_potential(cell, point, dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_DIVERGENCECLEANINGSOURCE_IMPL_HPP
