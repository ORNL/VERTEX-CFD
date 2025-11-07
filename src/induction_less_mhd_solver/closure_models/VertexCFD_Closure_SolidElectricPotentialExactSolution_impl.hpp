#ifndef VERTEXCFD_CLOSURE_SOLIDELECTRICPOTENTIALEXACTSOLUTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_SOLIDELECTRICPOTENTIALEXACTSOLUTION_IMPL_HPP

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
SolidElectricPotentialExactSolution<EvalType, Traits, NumSpaceDim>::
    SolidElectricPotentialExactSolution(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params)
    : _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _electric_potential("Exact_electric_potential", ir.dl_scalar)
    , _ir_degree(ir.cubature_degree)
{
    // Add evaluated field
    this->addEvaluatedField(_lagrange_pressure);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");
    this->addEvaluatedField(_electric_potential);

    // Get heat source value and coefficient for exact solution
    _sigma = closure_params.get<double>("Electrical Conductivity Coefficient");
    _phi_right = closure_params.get<double>("Right Electric Boundary Value");
    _phi_left = closure_params.get<double>("Left Electric Boundary Value");

    // Set name
    this->setName("Conduction Exact Solution");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void SolidElectricPotentialExactSolution<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void SolidElectricPotentialExactSolution<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
SolidElectricPotentialExactSolution<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _electric_potential.extent(1);
    using Kokkos::exp;
    using Kokkos::log;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // NS variables are still defined on a solid
            // region because they are expected by the Tempus
            // observer that compute the error norms.
            _lagrange_pressure(cell, point) = 0.0;
            for (int dim = 0; dim < num_space_dim; ++dim)
                _velocity[dim](cell, point) = 0.0;

            // With electric_potential-dependent sigma
            const double x = _ip_coords(cell, point, 0);
            _electric_potential(cell, point)
                = _phi_left * exp(log(_phi_right / _phi_left) * x);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_SOLIDELECTRICPOTENTIALEXACTSOLUTION_IMPL_HPP
