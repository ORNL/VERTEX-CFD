#ifndef VERTEXCFD_CLOSURE_HARTMANNPROBLEMEXACT_IMPL_HPP
#define VERTEXCFD_CLOSURE_HARTMANNPROBLEMEXACT_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include "Panzer_GlobalIndexer.hpp"
#include "Panzer_PureBasis.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
HartmannProblemExact<EvalType, Traits, NumSpaceDim>::HartmannProblemExact(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params,
    const Teuchos::ParameterList& user_params)
    : _exact_lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _exact_elec_pot("Exact_electric_potential", ir.dl_scalar)
    , _nu(closure_params.get<double>("Kinematic Viscosity"))
    , _sigma(closure_params.get<double>("Electrical Conductivity"))
    , _rho(closure_params.get<double>("Density"))
    , _L(closure_params.get<double>("Hartmann Solution Half-Width Channel"))
    , _B(0.0)
    , _ir_degree(ir.cubature_degree)
{
    // Evaluated fields
    this->addEvaluatedField(_exact_lagrange_pressure);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _exact_velocity, "Exact_velocity_");
    this->addEvaluatedField(_exact_elec_pot);

    // Get external magnetic vector
    const auto ext_magn_vct
        = user_params.get<Teuchos::Array<double>>("External Magnetic Field");
    for (int dim = 0; dim < 3; ++dim)
    {
        _B += ext_magn_vct[dim] * ext_magn_vct[dim];
    }
    _B = std::sqrt(_B);

    // Hartmann Number
    _Ha = _B * _L * sqrt(_sigma / (_rho * _nu));

    // Closure model name
    this->setName("Exact Solution Hartmann Problem");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void HartmannProblemExact<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void HartmannProblemExact<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
HartmannProblemExact<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _exact_elec_pot.extent(1);
    using Kokkos::cosh;
    using Kokkos::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Note: an analytical solution is only available for
            // the velocity and the electric potential.
            _exact_lagrange_pressure(cell, point) = 0.0;
            const double y = _ip_coords(cell, point, 1);

            _exact_velocity[0](cell, point) = (cosh(_Ha) - cosh(_Ha * y / _L))
                                              / (cosh(_Ha) - 1.0);
            _exact_velocity[1](cell, point) = 0.0;
            if (num_space_dim == 3)
                _exact_velocity[2](cell, point) = 0.0;
            _exact_elec_pot(cell, point) = 0.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_HARTMANNPROBLEMEXACT_IMPL_HPP
