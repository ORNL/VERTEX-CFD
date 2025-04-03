#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEERRORNORMS_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEERRORNORMS_IMPL_HPP

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
IncompressibleErrorNorms<EvalType, Traits, NumSpaceDim>::IncompressibleErrorNorms(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& user_params)
    : _L1_error_continuity("L1_Error_continuity", ir.dl_scalar)
    , _L1_error_energy("L1_Error_energy", ir.dl_scalar)
    , _L2_error_continuity("L2_Error_continuity", ir.dl_scalar)
    , _L2_error_energy("L2_Error_energy", ir.dl_scalar)
    , _volume("volume", ir.dl_scalar)
    , _exact_lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _exact_temperature("Exact_temperature", ir.dl_scalar)
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _temperature("temperature", ir.dl_scalar)
{
    // Temperature boolean
    _use_temp = user_params.isType<bool>("Build Temperature Equation")
                    ? user_params.get<bool>("Build Temperature Equation")
                    : false;

    // exact solution
    this->addDependentField(_exact_lagrange_pressure);
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _exact_velocity, "Exact_velocity_");
    if (_use_temp)
        this->addDependentField(_exact_temperature);

    // numerical solution
    this->addDependentField(_lagrange_pressure);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    if (_use_temp)
        this->addDependentField(_temperature);

    // error between exact and numerical solution
    this->addEvaluatedField(_L1_error_continuity);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _L1_error_momentum, "L1_Error_momentum_");
    this->addEvaluatedField(_L2_error_continuity);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _L2_error_momentum, "L2_Error_momentum_");

    if (_use_temp)
    {
        this->addEvaluatedField(_L1_error_energy);
        this->addEvaluatedField(_L2_error_energy);
    }
    this->addEvaluatedField(_volume);

    this->setName("Incompressible Error Norms " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleErrorNorms<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleErrorNorms<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::abs;
            using std::pow;

            // L1/L2 error norms
            _L1_error_continuity(cell, point)
                = abs(_lagrange_pressure(cell, point)
                      - _exact_lagrange_pressure(cell, point));

            if (_use_temp)
            {
                _L1_error_energy(cell, point)
                    = abs(_temperature(cell, point)
                          - _exact_temperature(cell, point));
            }

            _L2_error_continuity(cell, point)
                = pow(_lagrange_pressure(cell, point)
                          - _exact_lagrange_pressure(cell, point),
                      2);

            if (_use_temp)
            {
                _L2_error_energy(cell, point) = pow(
                    _temperature(cell, point) - _exact_temperature(cell, point),
                    2);
            }

            for (int i = 0; i < num_space_dim; ++i)
            {
                _L1_error_momentum[i](cell, point)
                    = abs(_velocity[i](cell, point)
                          - _exact_velocity[i](cell, point));
                _L2_error_momentum[i](cell, point) = pow(
                    _velocity[i](cell, point) - _exact_velocity[i](cell, point),
                    2);
            }

            _volume(cell, point) = 1.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEERRORNORMS_IMPL_HPP
