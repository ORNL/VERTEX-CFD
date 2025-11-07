#ifndef VERTEXCFD_INITIALCONDITION_SINUSOIDAL_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_SINUSOIDAL_IMPL_HPP

#include <Kokkos_MathematicalFunctions.hpp>
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
Sinusoidal<EvalType, Traits>::Sinusoidal(const Teuchos::ParameterList& params,
                                         const panzer::PureBasis& basis)
    : _basis_name(basis.name())
    , _basis_dim(basis.dimension())
    , _ic(_basis_name, basis.functional)
{
    const std::string dof_name = params.get<std::string>("Equation Set Name");
    _amplitude = params.get<double>("Amplitude");
    _phase = params.get<double>("Phase Angle");

    this->addEvaluatedField(_ic);
    this->addUnsharedField(_ic.fieldTag().clone());
    this->setName("Sinusoidal Initial Condition: " + dof_name);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void Sinusoidal<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void Sinusoidal<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void Sinusoidal<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _ic.extent(1);
    using Kokkos::sin;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_basis),
                         [&](const int basis) {
                             double sum = 0.0;
                             for (int dim = 0; dim < _basis_dim; ++dim)
                             {
                                 sum += _basis_coords(cell, basis, dim);
                             }
                             _ic(cell, basis) = _amplitude * sin(sum + _phase);
                         });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_SINUSOIDAL_IMPL_HPP
