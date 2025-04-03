#ifndef VERTEXCFD_INITIALCONDITION_CIRCLE_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_CIRCLE_IMPL_HPP

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
template<class EvalType, class Traits, int NumSpaceDim>
Circle<EvalType, Traits, NumSpaceDim>::Circle(
    const Teuchos::ParameterList& params, const panzer::PureBasis& basis)
    : _basis_name(basis.name())
{
    _inside_value = params.get<double>("Inside Value");
    _outside_value = params.get<double>("Outside Value");
    auto origin = params.get<Teuchos::Array<double>>("Center");
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        _origin[dim] = origin[dim];
    }
    _radius = params.get<double>("Radius");
    std::string dof_name = params.get<std::string>("Equation Set Name");
    _ic = PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>(
        dof_name, basis.functional);
    this->addEvaluatedField(_ic);
    this->addUnsharedField(_ic.fieldTag().clone());
    this->setName("Circle Initial Condition: " + dof_name);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void Circle<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void Circle<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void Circle<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _ic.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            using std::sqrt;
            double sum = 0.0;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const double coord = _basis_coords(cell, basis, dim)
                                     - _origin[dim];
                sum += coord * coord;
            }
            const double r = sqrt(sum);
            _ic(cell, basis) = r < _radius ? _inside_value : _outside_value;
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_CIRCLE_IMPL_HPP
