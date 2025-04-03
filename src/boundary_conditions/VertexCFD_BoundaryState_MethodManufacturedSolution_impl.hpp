#ifndef VERTEXCFD_BOUNDARYSTATE_METHODMANUFACTUREDSOLUTION_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_METHODMANUFACTUREDSOLUTION_IMPL_HPP

#include "utils/VertexCFD_Utils_Constants.hpp"
#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
MethodManufacturedSolution<EvalType, Traits, NumSpaceDim>::MethodManufacturedSolution(
    const panzer::IntegrationRule& ir)
    : _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _ir_degree(ir.cubature_degree)
{
    // Add evaludated fields
    this->addEvaluatedField(_boundary_lagrange_pressure);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _boundary_velocity, "BOUNDARY_velocity_");
    this->addEvaluatedField(_boundary_temperature);

    this->addEvaluatedField(_boundary_grad_lagrange_pressure);
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_vector,
                                   _boundary_grad_velocity,
                                   "BOUNDARY_GRAD_velocity_");
    this->addEvaluatedField(_boundary_grad_temperature);

    this->setName("Boundary State Method Manufactured Solution "
                  + std::to_string(num_space_dim) + "D");

    // Initialize coefficients for MMS
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
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MethodManufacturedSolution<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
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
    const int num_point = _boundary_lagrange_pressure.extent(1);
    using Constants::pi;
    using std::cos;
    using std::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // function = B + A * sin(2*pi*f_x*(x-phi_x)) *
            // sin(2*pi*f_y*(x-phi_y))
            //                  * sin(2*pi*f_z*(z-phi_z)) for 3D

            auto set_function
                = [=](const Kokkos::Array<double, num_coeff> coeff,
                      const Kokkos::Array<double, num_space_dim> coords) {
                      double return_val = coeff[0];
                      for (int i = 0; i < num_space_dim; ++i)
                          return_val
                              *= sin(2.0 * pi * coeff[2 * (i + 1)]
                                     * (coords[i] - coeff[2 * (i + 1) + 1]));
                      return_val += coeff[1];
                      return return_val;
                  };

            // function = A * 2*pi*f_x* cos(2*pi*f_x*(x-phi_x)) *
            // sin(2*pi*f_y*(y-phi_y))
            //              * sin(2*pi*f_z*(z-phi_z)) for 3D
            auto set_gradX_function =
                [=](const Kokkos::Array<double, num_coeff> coeff,
                    const Kokkos::Array<double, num_space_dim> coords) {
                    double return_val
                        = coeff[0] * 2.0 * pi * coeff[2]
                          * cos(2.0 * pi * coeff[2] * (coords[0] - coeff[3]))
                          * sin(2.0 * pi * coeff[4] * (coords[1] - coeff[5]));
                    if (num_space_dim == 3)
                        return_val *= sin(2.0 * pi * coeff[6]
                                          * (coords[2] - coeff[7]));
                    return return_val;
                };

            // function = A * sin(2*pi*f_x*(x-phi_x)) * 2*pi*f_y*
            // cos(2*pi*f_y*(y-phi_y))
            //              * sin(2*pi*f_z*(z-phi_z)) for 3D
            auto set_gradY_function =
                [=](const Kokkos::Array<double, num_coeff> coeff,
                    const Kokkos::Array<double, num_space_dim> coords) {
                    double return_val
                        = coeff[0] * 2.0 * pi * coeff[4]
                          * sin(2.0 * pi * coeff[2] * (coords[0] - coeff[3]))
                          * cos(2.0 * pi * coeff[4] * (coords[1] - coeff[5]));
                    if (num_space_dim == 3)
                        return_val *= sin(2.0 * pi * coeff[6]
                                          * (coords[2] - coeff[7]));
                    return return_val;
                };

            // function = A * sin(2*pi*f_x*(x-phi_x)) * sin(2*pi*f_y*(y-phi_y))
            //              * 2*pi*f_Z * cos(2*pi*f_z*(z-phi_z)) for 3D
            auto set_gradZ_function =
                [=](const Kokkos::Array<double, num_coeff> coeff,
                    const Kokkos::Array<double, num_space_dim> coords) {
                    return coeff[0] * 2.0 * pi * coeff[6]
                           * sin(2.0 * pi * coeff[2] * (coords[0] - coeff[3]))
                           * sin(2.0 * pi * coeff[4] * (coords[1] - coeff[5]))
                           * cos(2.0 * pi * coeff[6] * (coords[2] - coeff[7]));
                };

            Kokkos::Array<double, num_space_dim> x;
            for (int dim = 0; dim < num_space_dim; ++dim)
                x[dim] = _ip_coords(cell, point, dim);

            const double phi = set_function(_phi_coeff, x);
            const double T = set_function(_T_coeff, x);

            _boundary_lagrange_pressure(cell, point) = phi;
            Kokkos::Array<scalar_type, num_space_dim> vel;
            for (int i = 0; i < num_space_dim; ++i)
            {
                _boundary_velocity[i](cell, point)
                    = set_function(_vel_coeff[i], x);
                vel[i] = _boundary_velocity[i](cell, point);
            }
            _boundary_temperature(cell, point) = T;

            _boundary_grad_lagrange_pressure(cell, point, 0)
                = set_gradX_function(_phi_coeff, x);
            _boundary_grad_lagrange_pressure(cell, point, 1)
                = set_gradY_function(_phi_coeff, x);

            _boundary_grad_velocity[0](cell, point, 0)
                = set_gradX_function(_vel_coeff[0], x);
            _boundary_grad_velocity[0](cell, point, 1)
                = set_gradY_function(_vel_coeff[0], x);
            _boundary_grad_velocity[1](cell, point, 0)
                = set_gradX_function(_vel_coeff[1], x);
            _boundary_grad_velocity[1](cell, point, 1)
                = set_gradY_function(_vel_coeff[1], x);

            _boundary_grad_temperature(cell, point, 0)
                = set_gradX_function(_T_coeff, x);
            _boundary_grad_temperature(cell, point, 1)
                = set_gradY_function(_T_coeff, x);

            if (num_space_dim == 3)
            {
                _boundary_grad_lagrange_pressure(cell, point, 2)
                    = set_gradZ_function(_phi_coeff, x);

                _boundary_grad_velocity[0](cell, point, 2)
                    = set_gradZ_function(_vel_coeff[0], x);
                _boundary_grad_velocity[1](cell, point, 2)
                    = set_gradZ_function(_vel_coeff[1], x);
                _boundary_grad_velocity[2](cell, point, 0)
                    = set_gradX_function(_vel_coeff[2], x);
                _boundary_grad_velocity[2](cell, point, 1)
                    = set_gradY_function(_vel_coeff[2], x);
                _boundary_grad_velocity[2](cell, point, 2)
                    = set_gradZ_function(_vel_coeff[2], x);

                _boundary_grad_temperature(cell, point, 2)
                    = set_gradZ_function(_T_coeff, x);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_METHODMANUFACTUREDSOLUTION_IMPL_HPP
