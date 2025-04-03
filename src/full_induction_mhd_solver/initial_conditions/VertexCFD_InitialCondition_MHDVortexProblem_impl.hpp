#ifndef VERTEXCFD_INITIALCONDITION_MHDVORTEXPROBLEM_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_MHDVORTEXPROBLEM_IMPL_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>
#include <utils/VertexCFD_Utils_VectorField.hpp>

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
MHDVortexProblem<EvalType, Traits, NumSpaceDim>::MHDVortexProblem(
    const Teuchos::ParameterList& params, const panzer::PureBasis& basis)
    : _lagrange_pressure("lagrange_pressure", basis.functional)
    , _basis_name(basis.name())
{
    const auto vel_0 = params.get<Teuchos::Array<double>>("velocity_0");
    const auto xy_0 = params.get<Teuchos::Array<double>>("center_0");
    for (int dim = 0; dim < 2; ++dim)
    {
        _vel_0[dim] = vel_0[dim];
        _xy_0[dim] = xy_0[dim];
    }

    this->addEvaluatedField(_lagrange_pressure);
    this->addUnsharedField(_lagrange_pressure.fieldTag().clone());

    Utils::addEvaluatedVectorField(
        *this, basis.functional, _velocity, "velocity_", true);

    Utils::addEvaluatedVectorField(*this,
                                   basis.functional,
                                   _induced_magnetic_field,
                                   "induced_magnetic_field_",
                                   true);

    this->setName("MHDVortexProblem " + std::to_string(num_space_dim)
                  + "D Initial Condition");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MHDVortexProblem<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MHDVortexProblem<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
MHDVortexProblem<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _lagrange_pressure.extent(1);
    using std::exp;
    using std::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            const double x = _basis_coords(cell, basis, 0);
            const double y = _basis_coords(cell, basis, 1);
            const double r2 = (x - _xy_0[0]) * (x - _xy_0[0])
                              + (y - _xy_0[1]) * (y - _xy_0[1]);

            _lagrange_pressure(cell, basis)
                = 1.0 + 0.5 * exp(1.) * (1. - r2 * exp(-r2));

            _induced_magnetic_field[0](cell, basis) = exp(0.5 * (1.0 - r2))
                                                      * (_xy_0[1] - y);
            _induced_magnetic_field[1](cell, basis) = exp(0.5 * (1.0 - r2))
                                                      * (x - _xy_0[0]);

            _velocity[0](cell, basis) = _induced_magnetic_field[0](cell, basis)
                                        + _vel_0[0];
            _velocity[1](cell, basis) = _induced_magnetic_field[1](cell, basis)
                                        + _vel_0[1];
            if (num_space_dim > 2)
            {
                _induced_magnetic_field[2](cell, basis) = 0.0;
                _velocity[2](cell, basis) = 0.0;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_MHDVORTEXPROBLEM_IMPL_HPP
