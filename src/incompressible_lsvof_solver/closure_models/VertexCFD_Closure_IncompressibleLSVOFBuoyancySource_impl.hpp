#ifndef VERTEXCFD_CLOSURE_BUOYANCYSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_BUOYANCYSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFBuoyancySource<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFBuoyancySource(const panzer::IntegrationRule& ir,
                                      const Teuchos::ParameterList& user_params,
                                      const std::string& field_prefix)
    : _rho(field_prefix + "density", ir.dl_scalar)
{
    const auto gravity = user_params.get<Teuchos::Array<double>>("Gravity");
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        _gravity[dim] = gravity[dim];
    }

    // Evaluated fields
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _buoyancy_momentum_source,
                                   "LSVOF_BUOYANCY_SOURCE_"
                                   "momentum_");

    // Dependent fields
    this->addDependentField(_rho);

    this->setName("Incompressible LSVOF Buoyancy Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFBuoyancySource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFBuoyancySource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _buoyancy_momentum_source[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
            {
                _buoyancy_momentum_source[mom_dim](cell, point)
                    = _rho(cell, point) * _gravity[mom_dim];
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_BUOYANCYSOURCE_IMPL_HPP
