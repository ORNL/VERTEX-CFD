#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleSSTEddyViscosity<EvalType, Traits, NumSpaceDim>::
    IncompressibleSSTEddyViscosity(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& turb_params,
                                   bool on_wall_boundary)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _grad_turb_kinetic_energy("GRAD_turb_kinetic_energy", ir.dl_vector)
    , _grad_turb_specific_dissipation_rate(
          "GRAD_turb_specific_dissipation_rate", ir.dl_vector)
    , _distance("distance", ir.dl_scalar)
    , _rho("density", ir.dl_scalar)
    , _nu("kinematic_viscosity", ir.dl_scalar)
    , _a_1(0.31)
    , _beta_star(0.09)
    , _sigma_w2(0.856)
    , _on_wall_boundary(on_wall_boundary)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _sst_blending_function("sst_blending_function", ir.dl_scalar)
{
    // Check for user-defined coefficients or parameters
    if (turb_params.isType<double>("beta_star"))
        _beta_star = turb_params.get<double>("beta_star");

    if (turb_params.isType<double>("a_1"))
        _a_1 = turb_params.get<double>("a_1");

    if (turb_params.isType<double>("sigma_w2"))
        _sigma_w2 = turb_params.get<double>("sigma_w2");

    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);
    this->addDependentField(_grad_turb_kinetic_energy);
    this->addDependentField(_grad_turb_specific_dissipation_rate);
    if (!_on_wall_boundary)
    {
        this->addDependentField(_distance);
    }
    this->addDependentField(_rho);
    this->addDependentField(_nu);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_sst_blending_function);
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("Incompressible SST Turbulent Eddy Viscosity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSSTEddyViscosity<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_nu_t.get_view());
        bytes = scratch_view::shmem_size(_nu_t.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_nu_t.extent(1), NUM_TMPS);
    }
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleSSTEddyViscosity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using Kokkos::pow;
    using Kokkos::sqrt;
    using Kokkos::tanh;

    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-10;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_nu_t.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& vort_mag_x_F2 = _nu_t(cell, point);
            vort_mag_x_F2 = 0.0;
            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = i + 1; j < num_space_dim; ++j)
                {
                    vort_mag_x_F2
                        += pow(_grad_velocity[i](cell, point, j)
                                   - _grad_velocity[j](cell, point, i),
                               2.0);
                }
            }
            // The SmoothMath::max on the vorticity magnitude was not needed
            // with Trilinos 13, but with Trilinos 16 it will produce nans
            // without the protection
            vort_mag_x_F2 = sqrt(SmoothMath::max(vort_mag_x_F2, max_tol, 0.0));

            // This was also not needed with Trilinos 13, but is needed in
            // Trilinos 16 to survive negative tke values that crop up early in
            // the calculation for the 2d channel case
            auto&& tke = tmp_field(point, TKE);
            tke = SmoothMath::max(
                _turb_kinetic_energy(cell, point), max_tol, 0.0);
            auto&& omega = tmp_field(point, OMEGA);
            omega = SmoothMath::max(
                _turb_specific_dissipation_rate(cell, point), max_tol, 0.0);

            _sst_blending_function(cell, point) = 1.0;

            if (!_on_wall_boundary)
            {
                if (_distance(cell, point) > max_tol)
                {
                    // Compute the positive part of the cross diffusion term
                    // (dkdxj_domegadxj)
                    auto&& C = _sst_blending_function(cell, point);
                    C = 0.0;
                    for (int j = 0; j < num_space_dim; ++j)
                    {
                        C += _grad_turb_kinetic_energy(cell, point, j)
                             * _grad_turb_specific_dissipation_rate(
                                 cell, point, j);
                    }
                    // The 1.0e-20 limiter in the CD_kw calculation below is a
                    // model parameter that is modified in the 2003 revision of
                    // the model.
                    C = SmoothMath::max(
                        2.0 * _rho(cell, point) * _sigma_w2 * C / omega,
                        1.0e-20,
                        0.0);

                    // The original reference and the NASA write-up don't use
                    // these temporaries, but it should help us avoid
                    // recalculation. The meanings/uses of the terms are given
                    // below (according to Menter (1994).

                    // arg1 = min( max( A, B ), C )
                    // arg2 = max( 2A, B )

                    // A is the turbulent length scale divided by the wall
                    // distance. It goes to zero at the wall, is constant in
                    // the boundary layer log region, and goes to zero at the
                    // edge of the boundary layer.
                    auto&& A = tmp_field(point, LOG_REGION);
                    A = sqrt(tke)
                        / (_beta_star * omega * _distance(cell, point));

                    // B ensures that F_1 (the blending function) is equal to 1
                    // in the sublayer. omega goes like 1/d^2 near the wall and
                    // like 1/d in the log region so the B term is constant
                    // near the wall and goes to zero in the log region.
                    auto&& B = tmp_field(point, NEAR_WALL);
                    B = 500.0 * _nu(cell, point)
                        / (omega * pow(_distance(cell, point), 2));

                    // C is an additional safeguard against a
                    // freestream-dependent solution. According to Menter
                    // (1994) it ensures that arg1 goes to zero near the
                    // boundary layer edge.
                    C = 4.0 * _rho(cell, point) * _sigma_w2 * tke
                        / (C * pow(_distance(cell, point), 2));

                    // Compute F_1, arg1 is the min(max) expression in the pow
                    // call
                    _sst_blending_function(cell, point) = tanh(pow(
                        SmoothMath::min(SmoothMath::max(A, B, 0.0), C, 0.0), 4));

                    // Compute F_2 and multiply it with the vorticity
                    // magnitude, arg2 is the max expression in the pow call
                    vort_mag_x_F2
                        *= tanh(pow(SmoothMath::max(2.0 * A, B, 0.0), 2));
                }
            }

            // All of the terms are now protected against negative and zero
            // values, only one max needed
            _nu_t(cell, point)
                = _a_1 * tke
                  / SmoothMath::max(_a_1 * omega, vort_mag_x_F2, max_tol);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_IMPL_HPP
