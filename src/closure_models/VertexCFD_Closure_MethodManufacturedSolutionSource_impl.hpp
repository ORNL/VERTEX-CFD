#ifndef VERTEXCFD_CLOSURE_METHODMANUFACTUREDSOLUTIONSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_METHODMANUFACTUREDSOLUTIONSOURCE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Sacado.hpp>

#include <cmath>
#include <type_traits>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
MethodManufacturedSolutionSource<EvalType, Traits, NumSpaceDim>::
    MethodManufacturedSolutionSource(
        const panzer::IntegrationRule& ir,
        const bool build_viscous_flux,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _continuity_mms_source("MMS_SOURCE_continuity", ir.dl_scalar)
    , _energy_mms_source("MMS_SOURCE_energy", ir.dl_scalar)
    , _ir_degree(ir.cubature_degree)
    , _build_viscous_flux(build_viscous_flux)
    , _rho(fluid_prop.constantDensity())
    , _nu(fluid_prop.constantKinematicViscosity())
    , _kappa(0.0)
{
    this->addEvaluatedField(_continuity_mms_source);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _momentum_mms_source, "MMS_SOURCE_momentum_");
    this->addEvaluatedField(_energy_mms_source);

    // MMS function coefficients
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

    this->setName("Method of Manufactured Solution Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MethodManufacturedSolutionSource<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MethodManufacturedSolutionSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    Kokkos::RangePolicy<PHX::Device> policy(0, workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
template<typename T>
KOKKOS_INLINE_FUNCTION T
MethodManufacturedSolutionSource<EvalType, Traits, NumSpaceDim>::set_function(
    const Kokkos::Array<double, num_coeff>& coeff,
    const Kokkos::Array<T, num_space_dim>& x) const
{
    using Constants::pi;

    // Function = B+ A * sin(2*pi*f_x*(x-phi_x)) * sin(2*pi*f_y*(y-phi_y))
    //                 * sin(2*pi*f_z*(z-phi_z)) for 3D
    T val = coeff[0];
    for (int i = 0; i < num_space_dim; ++i)
        val *= sin(2.0 * pi * coeff[2 * (i + 1)]
                   * (x[i] - coeff[2 * (i + 1) + 1]));
    val += coeff[1];
    return val;
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
MethodManufacturedSolutionSource<EvalType, Traits, NumSpaceDim>::operator()(
    const int cell) const
{
    const int num_point = _continuity_mms_source.extent(1);
    using Constants::pi;
    const double third = 1.0 / 3.0;

    for (int point = 0; point < num_point; ++point)
    {
        using diff1_type = Sacado::Fad::SFad<scalar_type, num_space_dim>;
        using diff2_type = Sacado::Fad::SFad<diff1_type, num_space_dim>;

        // Create independent variables.
        Kokkos::Array<diff1_type, num_space_dim> diff1_x;
        Kokkos::Array<diff2_type, num_space_dim> diff2_x;

        // Assign values
        for (int i = 0; i < num_space_dim; ++i)
            diff1_x[i] = _ip_coords(cell, point, i);

        // Initialize derivative
        for (int i = 0; i < num_space_dim; ++i)
        {
            diff1_x[i].diff(i, num_space_dim);
            diff2_x[i] = diff1_x[i];
        }

        for (int i = 0; i < num_space_dim; ++i)
            diff2_x[i].diff(i, num_space_dim);

        // Variables with second derivatives
        Kokkos::Array<diff2_type, num_space_dim> diff2_vel;
        for (int i = 0; i < num_space_dim; ++i)
            diff2_vel[i] = set_function(_vel_coeff[i], diff2_x);
        const diff2_type diff2_T = set_function(_T_coeff, diff2_x);

        // diff1 gradients
        Kokkos::Array<Kokkos::Array<diff1_type, num_space_dim>, num_space_dim>
            diff1_grad_u;
        Kokkos::Array<diff1_type, num_space_dim> diff1_grad_T;
        for (int i = 0; i < num_space_dim; ++i)
        {
            for (int j = 0; j < num_space_dim; ++j)
                diff1_grad_u[i][j] = diff2_vel[i].fastAccessDx(j);

            diff1_grad_T[i] = diff2_T.fastAccessDx(i);
        }

        diff1_type diff1_tr_grad_u = 0.0;
        for (int i = 0; i < num_space_dim; ++i)
            diff1_tr_grad_u += diff1_grad_u[i][i];
        diff1_tr_grad_u *= third;

        // diff1 primitives
        const diff1_type diff1_phi = set_function(_phi_coeff, diff1_x);

        Kokkos::Array<diff1_type, num_space_dim> diff1_vel;
        for (int i = 0; i < num_space_dim; ++i)
            diff1_vel[i] = diff2_vel[i].val();
        const diff1_type diff1_T = diff2_T.val();

        //////////////////////////////
        // Convective flux
        //////////////////////////////
        Kokkos::Array<Kokkos::Array<diff1_type, num_space_dim>, num_conserve>
            conv_flux;

        // Convective continuity
        for (int i = 0; i < num_space_dim; ++i)
            conv_flux[0][i] = diff1_vel[i];

        // Convective momentum
        for (int i = 0; i < num_space_dim; ++i)
            for (int j = 0; j < num_space_dim; ++j)
            {
                conv_flux[i + 1][j] = diff1_vel[i] * diff1_vel[j];
                if (i == j)
                    conv_flux[i + 1][j] += diff1_phi;
            }

        // Convective energy
        for (int i = 0; i < num_space_dim; ++i)
        {
            conv_flux[num_conserve - 1][i] = diff1_T * diff1_vel[i];
        }

        // Total flux
        Kokkos::Array<Kokkos::Array<diff1_type, num_space_dim>, num_conserve> flux;
        for (int i = 0; i < num_conserve; ++i)
            for (int j = 0; j < num_space_dim; ++j)
                flux[i][j] = conv_flux[i][j];

        //////////////////////////////
        // Vicous flux
        //////////////////////////////
        if (_build_viscous_flux)
        {
            // Compute viscous stress tensor
            Kokkos::Array<Kokkos::Array<diff1_type, num_space_dim>, num_space_dim>
                diff1_tau;
            for (int i = 0; i < num_space_dim; ++i)
                for (int j = 0; j < num_space_dim; ++j)
                {
                    if (i == j)
                        diff1_tau[i][j]
                            = 2.0 * _nu
                              * (diff1_grad_u[i][j] - diff1_tr_grad_u);
                    else
                        diff1_tau[i][j]
                            = _nu * (diff1_grad_u[i][j] + diff1_grad_u[j][i]);
                }

            // Vicous flux
            Kokkos::Array<Kokkos::Array<diff1_type, num_space_dim>, num_conserve>
                visc_flux;

            // Viscous continuity
            for (int i = 0; i < num_space_dim; ++i)
                visc_flux[0][i] = 0.0;

            // Viscous momentum
            for (int i = 0; i < num_space_dim; ++i)
                for (int j = 0; j < num_space_dim; ++j)
                    visc_flux[i + 1][j] = diff1_tau[i][j];

            // Viscous energy
            for (int i = 0; i < num_space_dim; ++i)
            {
                diff1_type viscous_work = 0.0;
                for (int j = 0; j < num_space_dim; ++j)
                    viscous_work += diff1_vel[j] * diff1_tau[i][j];

                visc_flux[num_conserve - 1][i] = viscous_work
                                                 + _kappa * diff1_grad_T[i];
            }

            // Total flux
            for (int i = 0; i < num_conserve; ++i)
                for (int j = 0; j < num_space_dim; ++j)
                    flux[i][j] -= visc_flux[i][j];
        }

        // Sum up flux in each direction
        scalar_type continuity_sum = 0.0;
        Kokkos::Array<scalar_type, num_space_dim> mom_sum;
        for (int i = 0; i < num_space_dim; ++i)
            mom_sum[i] = 0.0;
        scalar_type energy_sum = 0.0;
        for (int i = 0; i < num_space_dim; ++i)
        {
            continuity_sum += flux[0][i].fastAccessDx(i);
            for (int j = 0; j < num_space_dim; ++j)
                mom_sum[i] += flux[i + 1][j].fastAccessDx(j);
            // FIXME:  warning: iteration 2 invokes undefined behavior
            // [-Waggressive-loop-optimizations]
            energy_sum += flux[num_conserve - 1][i].fastAccessDx(i);
        }

        _continuity_mms_source(cell, point) = continuity_sum;
        for (int i = 0; i < num_space_dim; ++i)
            _momentum_mms_source[i](cell, point) = mom_sum[i];
        _energy_mms_source(cell, point) = energy_sum;
    }
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_METHODMANUFACTUREDSOLUTIONSOURCE_IMPL_HPP
