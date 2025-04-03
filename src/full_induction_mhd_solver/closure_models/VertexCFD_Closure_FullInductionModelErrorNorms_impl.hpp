#ifndef VERTEXCFD_CLOSURE_FULLINDUCTIONMODELERRORNORMS_IMPL_HPP
#define VERTEXCFD_CLOSURE_FULLINDUCTIONMODELERRORNORMS_IMPL_HPP

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
FullInductionModelErrorNorms<EvalType, Traits, NumSpaceDim>::
    FullInductionModelErrorNorms(const panzer::IntegrationRule& ir)
    : _volume("volume", ir.dl_scalar)
{
    // exact solution
    Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _exact_induced_magnetic_field,
                                   "Exact_induced_magnetic_field_");

    // numerical solution
    Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _induced_magnetic_field,
                                   "induced_magnetic_field_");

    // error between exact and numerical solution
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _L1_error_induced, "L1_Error_induction_");
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _L2_error_induced, "L2_Error_induction_");

    this->addEvaluatedField(_volume);

    this->setName("Full Induction Model Error Norms "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void FullInductionModelErrorNorms<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
FullInductionModelErrorNorms<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _induced_magnetic_field[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::abs;
            using std::pow;

            // L1/L2 error norms
            for (int i = 0; i < num_space_dim; ++i)
            {
                _L1_error_induced[i](cell, point)
                    = abs(_induced_magnetic_field[i](cell, point)
                          - _exact_induced_magnetic_field[i](cell, point));
                _L2_error_induced[i](cell, point)
                    = pow(_induced_magnetic_field[i](cell, point)
                              - _exact_induced_magnetic_field[i](cell, point),
                          2);
            }

            _volume(cell, point) = 1.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_FULLINDUCTIONMODELERRORNORMS_IMPL_HPP
