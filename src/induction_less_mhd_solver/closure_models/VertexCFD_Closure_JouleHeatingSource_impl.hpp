#ifndef VERTEXCFD_CLOSURE_JOULEHEATINGSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_JOULEHEATINGSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
JouleHeatingSource<EvalType, Traits, NumSpaceDim>::JouleHeatingSource(
    const panzer::IntegrationRule& ir)
    : _jh_source("VOLUMETRIC_SOURCE_energy", ir.dl_scalar)
    , _sigma("electrical_conductivity", ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_jh_source);

    // Dependent fields
    this->addDependentField(_sigma);
    Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _electric_current_density,
                                   "electric_current_density_");

    this->setName("Joule Heating Source Term " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void JouleHeatingSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
JouleHeatingSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _jh_source.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Joule heating source term
            _jh_source(cell, point) = 0.0;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _jh_source(cell, point)
                    += _electric_current_density[dim](cell, point)
                       * _electric_current_density[dim](cell, point)
                       / _sigma(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_JOULEHEATINGSOURCE_IMPL_HPP
