#ifndef VERTEXCFD_INITIALCONDITION_STEP_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_STEP_IMPL_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
Step<EvalType, Traits>::Step(const Teuchos::ParameterList& params,
                             const panzer::PureBasis& basis)
    : _basis_name(basis.name())
{
    _left_value = params.get<double>("Left Value");
    _right_value = params.get<double>("Right Value");
    _origin = params.get<double>("Origin");
    std::string dof_name = params.get<std::string>("Equation Set Name");
    _ic = PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>(
        dof_name, basis.functional);
    this->addEvaluatedField(_ic);
    this->addUnsharedField(_ic.fieldTag().clone());
    this->setName("Step Initial Condition: " + dof_name);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void Step<EvalType, Traits>::postRegistrationSetup(typename Traits::SetupData sd,
                                                   PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void Step<EvalType, Traits>::evaluateFields(typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void Step<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _ic.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            const double x = _basis_coords(cell, basis, 0);
            _ic(cell, basis) = x < _origin ? _left_value : _right_value;
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_STEP_IMPL_HPP
