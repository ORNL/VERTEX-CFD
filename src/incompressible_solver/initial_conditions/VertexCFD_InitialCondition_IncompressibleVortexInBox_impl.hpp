#ifndef VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLEVORTEXINBOX_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLEVORTEXINBOX_IMPL_HPP

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
template<class EvalType, class Traits>
IncompressibleVortexInBox<EvalType, Traits>::IncompressibleVortexInBox(
    const panzer::PureBasis& basis)
    : _velocity_0("velocity_0", basis.functional)
    , _velocity_1("velocity_1", basis.functional)
    , _lagrange_pressure("lagrange_pressure", basis.functional)
    , _basis_name(basis.name())
{
    this->addEvaluatedField(_velocity_0);
    this->addEvaluatedField(_velocity_1);
    this->addEvaluatedField(_lagrange_pressure);
    this->addUnsharedField(_velocity_0.fieldTag().clone());
    this->addUnsharedField(_velocity_1.fieldTag().clone());
    this->addUnsharedField(_lagrange_pressure.fieldTag().clone());
    this->setName("Vertex In the Box Initial Condition");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleVortexInBox<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleVortexInBox<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleVortexInBox<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _velocity_0.extent(1);

    using Constants::pi;
    using std::cos;
    using std::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            const double x = _basis_coords(cell, basis, 0);
            const double y = _basis_coords(cell, basis, 1);
            _velocity_0(cell, basis) = -2.0 * cos(pi * y) * sin(pi * y)
                                       * sin(pi * x) * sin(pi * x);
            _velocity_1(cell, basis) = 2.0 * cos(pi * x) * sin(pi * x)
                                       * sin(pi * y) * sin(pi * y);
            _lagrange_pressure(cell, basis) = 0.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLEVORTEXINBOX_IMPL_HPP
