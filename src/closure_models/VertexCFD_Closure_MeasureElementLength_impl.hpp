#ifndef VERTEXCFD_CLOSURE_MEASUREELEMENTLENGTH_IMPL_HPP
#define VERTEXCFD_CLOSURE_MEASUREELEMENTLENGTH_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <algorithm>
#include <cmath>

/**
This class implements the logic to calculate the size of a mesh element
from the Jacobian matrix. The implementation follows the same logic as in
MFEM
(https://mfem.github.io/doxygen/html/classmfem_1_1Mesh.html#adde50e5a10f09b877c861d8371981c0c)
**/

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
MeasureElementLength<EvalType, Traits>::MeasureElementLength(
    const panzer::IntegrationRule& ir, const std::string& prefix)
    : _element_length(prefix + "element_length", ir.dl_vector)
    , _ir_degree(ir.cubature_degree)
{
    this->addEvaluatedField(_element_length);
    this->setName("Measure Element Length");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MeasureElementLength<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MeasureElementLength<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _cell_det = workset.int_rules[_ir_index]->jac_det;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void MeasureElementLength<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _element_length.extent(1);
    const int num_space_dim = _element_length.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::pow;
            // Compute element size
            double h = pow(_cell_det(cell, point), 1.0 / num_space_dim);

            // Set value of the element length (same value for all
            // directions)
            for (int i = 0; i < num_space_dim; i++)
            {
                _element_length(cell, point, i) = h;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_MEASUREELEMENTLENGTH_IMPL_HPP
