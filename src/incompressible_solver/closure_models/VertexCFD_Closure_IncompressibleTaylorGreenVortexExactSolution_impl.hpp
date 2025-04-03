#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLETAYLORGREENVORTEXEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLETAYLORGREENVORTEXEXACTSOLUTION_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include "Panzer_GlobalIndexer.hpp"
#include "Panzer_PureBasis.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleTaylorGreenVortexExactSolution<EvalType, Traits, NumSpaceDim>::
    IncompressibleTaylorGreenVortexExactSolution(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _ir_degree(ir.cubature_degree)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _Ft(0.0)
{
    this->addEvaluatedField(_lagrange_pressure);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");

    this->setName("Incompressible Taylor Green Vortex Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTaylorGreenVortexExactSolution<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTaylorGreenVortexExactSolution<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);

    using std::exp;
    _Ft = exp(-2.0 * _nu * workset.time);
    _time = workset.time;

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleTaylorGreenVortexExactSolution<EvalType, Traits, NumSpaceDim>::
operator()(const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);

    using std::cos;
    using std::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const double x = _ip_coords(cell, point, 0);
            const double y = _ip_coords(cell, point, 1);

            _lagrange_pressure(cell, point)
                = -0.25 * (cos(2.0 * x) + cos(2.0 * y)) * _Ft * _Ft;
            _velocity[0](cell, point) = cos(x) * sin(y) * _Ft;
            _velocity[1](cell, point) = -sin(x) * cos(y) * _Ft;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLETAYLORGREENVORTEXEXACTSOLUTION_IMPL_HPP
