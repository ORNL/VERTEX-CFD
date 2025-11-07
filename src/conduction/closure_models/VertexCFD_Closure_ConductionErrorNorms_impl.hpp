#ifndef VERTEXCFD_CLOSURE_CONDUCTIONERRORNORMS_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONERRORNORMS_IMPL_HPP

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
ConductionErrorNorms<EvalType, Traits, NumSpaceDim>::ConductionErrorNorms(
    const panzer::IntegrationRule& ir)
    : _L1_error_continuity("L1_Error_continuity", ir.dl_scalar)
    , _L1_error_energy("L1_Error_energy", ir.dl_scalar)
    , _L2_error_continuity("L2_Error_continuity", ir.dl_scalar)
    , _L2_error_energy("L2_Error_energy", ir.dl_scalar)
    , _volume("volume", ir.dl_scalar)
    , _exact_temperature("Exact_temperature", ir.dl_scalar)
    , _temperature("temperature", ir.dl_scalar)
{
    // exact solution
    this->addDependentField(_exact_temperature);

    // numerical solution
    this->addDependentField(_temperature);

    // error between exact and numerical solution
    this->addEvaluatedField(_L1_error_continuity);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _L1_error_momentum, "L1_Error_momentum_");
    this->addEvaluatedField(_L2_error_continuity);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _L2_error_momentum, "L2_Error_momentum_");

    this->addEvaluatedField(_L1_error_energy);
    this->addEvaluatedField(_L2_error_energy);
    this->addEvaluatedField(_volume);

    this->setName("Conduction Error Norms " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void ConductionErrorNorms<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
ConductionErrorNorms<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _temperature.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::abs;
            using std::pow;

            // L1/L2 error norms for temperature. All other variables are set
            // to zero on the solid region.
            _L1_error_energy(cell, point) = abs(
                _temperature(cell, point) - _exact_temperature(cell, point));

            _L2_error_energy(cell, point) = pow(
                _temperature(cell, point) - _exact_temperature(cell, point), 2);

            _L1_error_continuity(cell, point) = 0.0;

            _L2_error_continuity(cell, point) = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                _L1_error_momentum[i](cell, point) = 0.0;
                _L2_error_momentum[i](cell, point) = 0.0;
            }

            _volume(cell, point) = 1.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_CONDUCTIONERRORNORMS_IMPL_HPP
