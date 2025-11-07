#ifndef VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

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
ConductionExactSolution<EvalType, Traits, NumSpaceDim>::ConductionExactSolution(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _temperature("Exact_temperature", ir.dl_scalar)
    , _ir_degree(ir.cubature_degree)
{
    // Add evaluated field
    this->addEvaluatedField(_lagrange_pressure);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");
    this->addEvaluatedField(_temperature);

    // Get heat source value and coefficient for exact solution
    _q = closure_params.get<double>("Volumetric Heat Source Value");
    _k = closure_params.get<double>("Thermal Conductivity Coefficient");
    _T_right = closure_params.get<double>("Right Temperature Boundary Value");

    // Set name
    this->setName("Conduction Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ConductionExactSolution<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ConductionExactSolution<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
ConductionExactSolution<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _temperature.extent(1);
    using std::exp;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // NS variables are still defined on a solid region because they
            // are expected by the Tempus observer that compute the error
            // norms.
            _lagrange_pressure(cell, point) = 0.0;
            _velocity[0](cell, point) = 0.0;
            _velocity[1](cell, point) = 0.0;
            if (num_space_dim == 3)
                _velocity[2](cell, point) = 0.0;

            // With temperature-dependent k and variable heat source
            const double x = _ip_coords(cell, point, 0);
            _temperature(cell, point)
                = _T_right * exp(_q / (6.0 * _k) * (1.0 - x * x * x));
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_CONDUCTIONEXACTSOLUTION_IMPL_HPP
