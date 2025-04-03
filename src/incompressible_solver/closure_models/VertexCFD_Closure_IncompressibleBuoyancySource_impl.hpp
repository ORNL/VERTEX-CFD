#ifndef VERTEXCFD_CLOSURE_BUOYANCYSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_BUOYANCYSOURCE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleBuoyancySource<EvalType, Traits, NumSpaceDim>::
    IncompressibleBuoyancySource(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params)
    : _buoyancy_continuity_source("BUOYANCY_SOURCE_continuity", ir.dl_scalar)
    , _buoyancy_energy_source("BUOYANCY_SOURCE_energy", ir.dl_scalar)
    , _temperature("temperature", ir.dl_scalar)
{
    const auto gravity = user_params.get<Teuchos::Array<double>>("Gravity");
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        _gravity[dim] = gravity[dim];
    }

    // Evaluated fields
    this->addEvaluatedField(_buoyancy_continuity_source);
    this->addEvaluatedField(_buoyancy_energy_source);

    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _buoyancy_momentum_source,
                                   "BUOYANCY_SOURCE_"
                                   "momentum_");

    _beta_T = fluid_prop.expansionCoefficient();
    _T_ref = fluid_prop.referenceTemperature();

    // Dependent fields
    this->addDependentField(_temperature);

    this->setName("Incompressible Buoyancy Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleBuoyancySource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleBuoyancySource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _buoyancy_continuity_source.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _buoyancy_continuity_source(cell, point) = 0.0;
            _buoyancy_energy_source(cell, point) = 0.0;

            for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
            {
                _buoyancy_momentum_source[mom_dim](cell, point)
                    = -_beta_T * _gravity[mom_dim]
                      * (_temperature(cell, point) - _T_ref);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_BUOYANCYSOURCE_IMPL_HPP
