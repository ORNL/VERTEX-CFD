#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGEXACTSOLUTION_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleSUPGExactSolution<EvalType, Traits, NumSpaceDim>::
    IncompressibleSUPGExactSolution(const panzer::IntegrationRule& ir,
                                    const Teuchos::ParameterList& closure_params)
    : _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _temperature("Exact_temperature", ir.dl_scalar)
    , _ir_degree(ir.cubature_degree)
    , _D(closure_params.get<double>("Thermal diffusivity"))
    , _u(closure_params.get<double>("Uniform advection velocity"))
{
    this->addEvaluatedField(_lagrange_pressure);
    this->addEvaluatedField(_temperature);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");

    this->setName("Incompressible SUPG Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSUPGExactSolution<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSUPGExactSolution<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleSUPGExactSolution<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);

    using Kokkos::exp;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const double x = _ip_coords(cell, point, 0);

            // Pressure and velocity are set to zero because assumed uniform
            _lagrange_pressure(cell, point) = 0.0;
            _velocity[0](cell, point) = 0.0;
            _velocity[1](cell, point) = 0.0;
            _temperature(cell, point)
                = (x - (1.0 - exp(_u * x / _D)) / (1.0 - exp(_u / _D))) / _u;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESUPGEXACTSOLUTION_IMPL_HPP
