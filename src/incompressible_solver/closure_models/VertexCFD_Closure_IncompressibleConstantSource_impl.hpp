#ifndef VERTEXCFD_CLOSURE_CONSTANTSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONSTANTSOURCE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleConstantSource<EvalType, Traits, NumSpaceDim>::
    IncompressibleConstantSource(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params)
    : _continuity_source("CONSTANT_SOURCE_continuity", ir.dl_scalar)
    , _energy_source("CONSTANT_SOURCE_energy", ir.dl_scalar)
    , _solve_temp(fluid_prop.solveTemperature())
{
    const auto mom_input_source
        = user_params.get<Teuchos::Array<double>>("Momentum Source");
    for (int dim = 0; dim < num_space_dim; ++dim)
        _mom_input_source[dim] = mom_input_source[dim];

    // Evaluated fields
    this->addEvaluatedField(_continuity_source);

    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _momentum_source,
                                   "CONSTANT_SOURCE_"
                                   "momentum_");

    if (_solve_temp)
    {
        _energy_input_source = user_params.get<double>("Energy Source");
        this->addEvaluatedField(_energy_source);
    }

    // Dependent fields
    this->setName("Incompressible Constant Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleConstantSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleConstantSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _continuity_source.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _continuity_source(cell, point) = 0.0;

            for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
                _momentum_source[mom_dim](cell, point)
                    = _mom_input_source[mom_dim];
            if (_solve_temp)
                _energy_source(cell, point) = _energy_input_source;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONSTANTSOURCE_IMPL_HPP
