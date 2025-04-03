#ifndef VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETAYLORGREENVORTEX_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETAYLORGREENVORTEX_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleTaylorGreenVortex<EvalType, Traits, NumSpaceDim>::
    IncompressibleTaylorGreenVortex(const panzer::PureBasis& basis)
    : _lagrange_pressure("lagrange_pressure", basis.functional)
    , _basis_name(basis.name())
{
    this->addEvaluatedField(_lagrange_pressure);
    this->addUnsharedField(_lagrange_pressure.fieldTag().clone());

    Utils::addEvaluatedVectorField(
        *this, basis.functional, _velocity, "velocity_", true);

    this->setName("Taylor-Green Vortex Initial Condition");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTaylorGreenVortex<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTaylorGreenVortex<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleTaylorGreenVortex<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _lagrange_pressure.extent(1);

    using std::cos;
    using std::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            const double x = _basis_coords(cell, basis, 0);
            const double y = _basis_coords(cell, basis, 1);
            _lagrange_pressure(cell, basis) = -0.25
                                              * (cos(2.0 * x) + cos(2.0 * y));
            _velocity[0](cell, basis) = cos(x) * sin(y);
            _velocity[1](cell, basis) = -sin(x) * cos(y);
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETAYLORGREENVORTEX_IMPL_HPP
