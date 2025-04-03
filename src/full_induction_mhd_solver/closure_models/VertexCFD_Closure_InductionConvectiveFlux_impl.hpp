#ifndef VERTEXCFD_CLOSURE_INDUCTIONCONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INDUCTIONCONVECTIVEFLUX_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
InductionConvectiveFlux<EvalType, Traits, NumSpaceDim>::InductionConvectiveFlux(
    const panzer::IntegrationRule& ir,
    const MHDProperties::FullInductionMHDProperties& mhd_props,
    const std::string& flux_prefix,
    const std::string& field_prefix)
    : _magnetic_correction_potential_flux(
          flux_prefix + "CONVECTIVE_FLUX_magnetic_correction_potential",
          ir.dl_vector)
    , _scalar_magnetic_potential(field_prefix + "scalar_magnetic_potential",
                                 ir.dl_scalar)
    , _magnetic_pressure("magnetic_pressure", ir.dl_scalar)
    , _solve_magn_corr(mhd_props.buildMagnCorr())
    , _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
    , _c_h(mhd_props.hyperbolicDivergenceCleaningSpeed())
{
    // Contributed fields
    Utils::addContributedVectorField(*this, ir.dl_vector, _momentum_flux,
                                   flux_prefix + "CONVECTIVE_FLUX_"
                                                 "momentum_");

    // Evaluated fields
    this->addEvaluatedField(_magnetic_correction_potential_flux);

    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _induction_flux,
                                 flux_prefix + "CONVECTIVE_FLUX_"
                                               "induction_");

    if (_solve_magn_corr)
    {
        this->addEvaluatedField(_magnetic_correction_potential_flux);
    }

    // Dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _total_magnetic_field, "total_magnetic_field_");
    this->addDependentField(_magnetic_pressure);

    if (_solve_magn_corr)
    {
        this->addDependentField(_scalar_magnetic_potential);
    }

    this->setName("Induction Convective Flux " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void InductionConvectiveFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
InductionConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _induction_flux[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int flux_dim = 0; flux_dim < num_space_dim; ++flux_dim)
            {
                for (int vec_dim = 0; vec_dim < num_space_dim; ++vec_dim)
                {
                    _momentum_flux[vec_dim](cell, point, flux_dim)
                        -= _total_magnetic_field[flux_dim](cell, point)
                           * _total_magnetic_field[vec_dim](cell, point)
                           / _magnetic_permeability;
                    if (vec_dim != flux_dim)
                    {
                        // Set the off-diagonal flux terms for the induction
                        // equation.
                        _induction_flux[vec_dim](cell, point, flux_dim)
                            = _velocity[flux_dim](cell, point)
                                  * _total_magnetic_field[vec_dim](cell, point)
                              - _total_magnetic_field[flux_dim](cell, point)
                                    * _velocity[vec_dim](cell, point);
                    }
                }
                // Add the magnetic pressure contribution to momentum flux.
                _momentum_flux[flux_dim](cell, point, flux_dim)
                    += _magnetic_pressure(cell, point);
                // Set diagonal flux terms for the induction equation, which
                // are nonzero only when using divergence cleaning.
                if (_solve_magn_corr)
                {
                    _induction_flux[flux_dim](cell, point, flux_dim)
                        = _c_h * _scalar_magnetic_potential(cell, point);
                    _magnetic_correction_potential_flux(cell, point, flux_dim)
                        = _c_h * _total_magnetic_field[flux_dim](cell, point);
                }
                else
                {
                    _induction_flux[flux_dim](cell, point, flux_dim) = 0.0;
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INDUCTIONCONVECTIVEFLUX_IMPL_HPP
