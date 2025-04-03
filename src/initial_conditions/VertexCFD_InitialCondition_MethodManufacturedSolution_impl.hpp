#ifndef VERTEXCFD_INITIALCONDITION_METHODMANUFACTUREDSOLUTION_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_METHODMANUFACTUREDSOLUTION_IMPL_HPP

#include "utils/VertexCFD_Utils_Constants.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
MethodManufacturedSolution<EvalType, Traits, NumSpaceDim>::MethodManufacturedSolution(
    const panzer::PureBasis& basis)
    : _lagrange_pressure("lagrange_pressure", basis.functional)
    , _temperature("temperature", basis.functional)
    , _basis_name(basis.name())
{
    this->addEvaluatedField(_lagrange_pressure);
    this->addUnsharedField(_lagrange_pressure.fieldTag().clone());
    this->addEvaluatedField(_temperature);
    this->addUnsharedField(_temperature.fieldTag().clone());

    Utils::addEvaluatedVectorField(
        *this, basis.functional, _velocity, "velocity_", true);

    this->setName("MethodManufacturedSolution Initial Condition"
                  + std::to_string(num_space_dim) + "D");

    _phi_coeff[0] = 0.0125;
    _phi_coeff[1] = 1.0;
    _phi_coeff[2] = 0.25;
    _phi_coeff[3] = 0.5;
    _phi_coeff[4] = 0.125;
    _phi_coeff[5] = 0.0;

    _vel_coeff[0][0] = 0.0125;
    _vel_coeff[0][1] = 0.08;
    _vel_coeff[0][2] = 0.125;
    _vel_coeff[0][3] = 0.0;
    _vel_coeff[0][4] = 0.125;
    _vel_coeff[0][5] = 0.25;

    _vel_coeff[1][0] = 0.0375;
    _vel_coeff[1][1] = 1.125;
    _vel_coeff[1][2] = 0.25;
    _vel_coeff[1][3] = 0.0;
    _vel_coeff[1][4] = 0.375;
    _vel_coeff[1][5] = 0.5;

    _T_coeff[0] = 0.0625;
    _T_coeff[1] = 1.0;
    _T_coeff[2] = 0.375;
    _T_coeff[3] = 0.25;
    _T_coeff[4] = 0.25;
    _T_coeff[5] = 0.5;

    if (num_space_dim == 3)
    {
        _phi_coeff[6] = 0.375;
        _phi_coeff[7] = 1.0;

        _vel_coeff[0][6] = 0.25;
        _vel_coeff[0][7] = 0.0;

        _vel_coeff[1][6] = 0.25;
        _vel_coeff[1][7] = 0.5;

        _vel_coeff[2][0] = 0.025;
        _vel_coeff[2][1] = 0.0;
        _vel_coeff[2][2] = 0.125;
        _vel_coeff[2][3] = 1.0;
        _vel_coeff[2][4] = 0.25;
        _vel_coeff[2][5] = 0.25;
        _vel_coeff[2][6] = 0.25;
        _vel_coeff[2][7] = 0.25;

        _T_coeff[6] = 0.125;
        _T_coeff[7] = 0.5;
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MethodManufacturedSolution<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MethodManufacturedSolution<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
MethodManufacturedSolution<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _lagrange_pressure.extent(1);
    using Constants::pi;
    using std::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            // function = B + A * sin(2*pi*f_x*(x-phi_x)) *
            // sin(2*pi*f_y*(x-phi_y))
            //                  * sin(2*pi*f_z*(z-phi_z)) for 3D
            auto set_function
                = [=](const Kokkos::Array<double, num_coeff> coeff,
                      const Kokkos::Array<double, num_space_dim> x) {
                      double return_val = coeff[0];
                      for (int i = 0; i < num_space_dim; ++i)
                          return_val *= sin(2.0 * pi * coeff[2 * (i + 1)]
                                            * (x[i] - coeff[2 * (i + 1) + 1]));
                      return_val += coeff[1];
                      return return_val;
                  };

            // Note this block is all of `auto` type rather than `scalar_type`
            // because the values coming from `_basis_coords` are never going
            // to be FAD objects. By using `auto` instead of `double` we don't
            // impose a precision on the mesh positions.

            // FIXME: Kokkos::Array<auto, num_space_dim> is okay??
            Kokkos::Array<double, num_space_dim> x;
            for (int dim = 0; dim < num_space_dim; ++dim)
                x[dim] = _basis_coords(cell, basis, dim);

            _lagrange_pressure(cell, basis) = set_function(_phi_coeff, x);
            _temperature(cell, basis) = set_function(_T_coeff, x);
            for (int i = 0; i < num_space_dim; ++i)
                _velocity[i](cell, basis) = set_function(_vel_coeff[i], x);
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_METHODMANUFACTUREDSOLUTION_IMPL_HPP
