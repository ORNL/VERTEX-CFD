#ifndef VERTEXCFD_INITIALCONDITION_DIVERGENCEADVECTIONTEST_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_DIVERGENCEADVECTIONTEST_IMPL_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include "utils/VertexCFD_Utils_Constants.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
DivergenceAdvectionTest<EvalType, Traits, NumSpaceDim>::DivergenceAdvectionTest(
    const Teuchos::ParameterList& mhd_params, const panzer::PureBasis& basis)
    : _lagrange_pressure("lagrange_pressure", basis.functional)
    , _basis_name(basis.name())
    , _phi(6.0)
    , _r0(1.0 / std::sqrt(8.0))
    , _xy_0({0.0, 0.0, 0.0})
    , _vel({1.0, 1.0, 0.0})
{
    Teuchos::ParameterList params;
    if (mhd_params.isSublist("Divergence Advection Test"))
    {
        params = mhd_params.sublist("Divergence Advection Test");
    }

    if (params.isType<double>("Lagrange Pressure"))
    {
        _phi = params.get<double>("Lagrange Pressure");
    }

    if (params.isType<double>("Divergence Bubble Radius"))
    {
        _r0 = params.get<double>("Divergence Bubble Radius");
    }

    if (params.isType<Teuchos::Array<double>>("Divergence Bubble Center"))
    {
        const auto xy_0
            = params.get<Teuchos::Array<double>>("Divergence Bubble Center");
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            _xy_0[dim] = xy_0[dim];
        }
    }

    if (params.isType<Teuchos::Array<double>>("Velocity"))
    {
        const auto vel = params.get<Teuchos::Array<double>>("Velocity");
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            _vel[dim] = vel[dim];
        }
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

    this->setName("DivergenceAdvectionTest " + std::to_string(num_space_dim)
                  + "D Initial Condition");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void DivergenceAdvectionTest<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void DivergenceAdvectionTest<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
DivergenceAdvectionTest<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _lagrange_pressure.extent(1);
    using Constants::pi;
    using std::pow;
    using std::sqrt;

    const double b_coeff = 1.0 / sqrt(4.0 * pi);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            const double x = _basis_coords(cell, basis, 0);
            const double y = _basis_coords(cell, basis, 1);
            const double dx = x - _xy_0[0];
            const double dy = y - _xy_0[1];
            const double r = sqrt(dx * dx + dy * dy);

            _lagrange_pressure(cell, basis) = _phi;

            if (r < _r0)
            {
                _induced_magnetic_field[0](cell, basis)
                    = b_coeff * (pow(r / _r0, 8) - 2.0 * pow(r / _r0, 4) + 1);
            }
            else
            {
                _induced_magnetic_field[0](cell, basis) = 0.0;
            }
            _induced_magnetic_field[1](cell, basis) = 0.0;

            _velocity[0](cell, basis) = _vel[0];
            _velocity[1](cell, basis) = _vel[1];

            if (num_space_dim > 2)
            {
                // For now, set induced B_z for 3D cases to zero, so that for
                // both 2D and 3D we do the same thing (2D has no induced Bz
                // so we need to use the ExternalMagneticField closure in that
                // case until / unless we switch to always solving B_z
                // independent of mesh dimension)
                _induced_magnetic_field[2](cell, basis) = 0.0;
                _velocity[2](cell, basis) = _vel[2];
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_DIVERGENCEADVECTIONTEST_IMPL_HPP
