#ifndef VERTEXCFD_INITIALCONDITION_GAUSSIAN_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_GAUSSIAN_IMPL_HPP

#include "utils/VertexCFD_Utils_Constants.hpp"

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
Gaussian<EvalType, Traits, NumSpaceDim>::Gaussian(
    const Teuchos::ParameterList& params, const panzer::PureBasis& basis)
    : _basis_name(basis.name())
    , _a(Kokkos::ViewAllocateWithoutInitializing("Gaussian a"),
         basis.dimension())
    , _b(Kokkos::ViewAllocateWithoutInitializing("Gaussian b"),
         basis.dimension())
    , _c(Kokkos::ViewAllocateWithoutInitializing("Gaussian c"),
         basis.dimension())
{
    std::string dof_name = params.get<std::string>("Equation Set Name");
    _ic = PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>(
        dof_name, basis.functional);
    this->addEvaluatedField(_ic);
    this->addUnsharedField(_ic.fieldTag().clone());
    this->setName("Gaussian Initial Condition: " + dof_name);

    auto center = params.get<Teuchos::Array<double>>("Center");
    auto sigma = params.get<Teuchos::Array<double>>("Sigma");
    _d = params.get<double>("Base");
    const double sqrt2pi = std::sqrt(2.0 * Constants::pi);
    auto a_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _a);
    auto b_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _b);
    auto c_host = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _c);
    for (int dim = 0; dim < basis.dimension(); ++dim)
    {
        a_host(dim) = 1.0 / (sqrt2pi * sigma[dim]);
        b_host(dim) = center[dim];
        c_host(dim) = 0.5 / (sigma[dim] * sigma[dim]);
    }
    Kokkos::deep_copy(_a, a_host);
    Kokkos::deep_copy(_b, b_host);
    Kokkos::deep_copy(_c, c_host);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void Gaussian<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void Gaussian<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void Gaussian<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _ic.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            using std::exp;
            double result = 1.0;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const auto x = _basis_coords(cell, basis, dim);
                result *= _a(dim)
                          * exp((_b(dim) - x) * (x - _b(dim)) * _c(dim));
            }
            _ic(cell, basis) = result + _d;
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_GAUSSIAN_IMPL_HPP
