#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <math.h>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleKTauSource<EvalType, Traits, NumSpaceDim>::IncompressibleKTauSource(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& turb_params)
    : _nu("kinematic_viscosity", ir.dl_scalar)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _grad_turb_kinetic_energy("GRAD_turb_kinetic_energy", ir.dl_vector)
    , _grad_turb_specific_dissipation_rate(
          "GRAD_turb_specific_dissipation_rate", ir.dl_vector)
    , _beta_star(0.09)
    , _gamma(0.52)
    , _beta_0(0.0708)
    , _sigma_d(0.125)
    , _sigma_t(0.5)
    , _limit_production(true)
    , _limit_destruction(true)
    , _k_source("SOURCE_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_prod("PRODUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_dest("DESTRUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _t_source("SOURCE_turb_specific_dissipation_rate_equation", ir.dl_scalar)
    , _t_prod("PRODUCTION_turb_specific_dissipation_rate_equation",
              ir.dl_scalar)
    , _t_dest_nut(
          "DESTRUCTION_eddy_viscosity_turb_specific_dissipation_rate_equation",
          ir.dl_scalar)
    , _t_dest_nu(
          "DESTRUCTION_viscosity_turb_specific_dissipation_rate_equation",
          ir.dl_scalar)
    , _t_cross("CROSS_DIFFUSION_turb_specific_dissipation_rate_equation",
               ir.dl_scalar)
{
    // Check for user-defined coefficients or parameters
    if (turb_params.isType<double>("beta_star"))
        _beta_star = turb_params.get<double>("beta_star");

    if (turb_params.isType<double>("gamma"))
        _gamma = turb_params.get<double>("gamma");

    if (turb_params.isType<double>("beta_0"))
        _beta_0 = turb_params.get<double>("beta_0");

    if (turb_params.isType<double>("sigma_d"))
        _sigma_d = turb_params.get<double>("sigma_d");

    if (turb_params.isType<double>("sigma_t"))
        _sigma_t = turb_params.get<double>("sigma_t");

    if (turb_params.isType<bool>("Limit Production Term"))
        _limit_production = turb_params.get<bool>("Limit Production Term");

    if (turb_params.isType<bool>("Limit Destruction Term"))
        _limit_destruction = turb_params.get<bool>("Limit Destruction Term");

    // Add dependent fields
    this->addDependentField(_nu);
    this->addDependentField(_nu_t);
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);
    this->addDependentField(_grad_turb_kinetic_energy);
    this->addDependentField(_grad_turb_specific_dissipation_rate);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_k_source);
    this->addEvaluatedField(_k_prod);
    this->addEvaluatedField(_k_dest);
    this->addEvaluatedField(_t_source);
    this->addEvaluatedField(_t_prod);
    this->addEvaluatedField(_t_dest_nut);
    this->addEvaluatedField(_t_dest_nu);
    this->addEvaluatedField(_t_cross);

    // Closure model name
    this->setName("K-Tau Incompressible Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleKTauSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleKTauSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-10;
    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& Sij_Sij = _t_prod(cell, point);
            Sij_Sij = 0.0;
            for (int i = 0; i < num_space_dim; ++i)
            {
                Sij_Sij += pow(_grad_velocity[i](cell, point, i), 2.0);
                for (int j = i + 1; j < num_space_dim; ++j)
                {
                    Sij_Sij += 0.5
                               * pow(_grad_velocity[i](cell, point, j)
                                         + _grad_velocity[j](cell, point, i),
                                     2.0);
                }
            }

            // Turbulent kinetic energy terms
            _k_prod(cell, point) = 2.0 * _nu_t(cell, point) * Sij_Sij;

            _k_dest(cell, point)
                = _beta_star * _turb_kinetic_energy(cell, point)
                  / SmoothMath::max(
                      _turb_specific_dissipation_rate(cell, point),
                      max_tol,
                      0.0);

            if (_limit_production)
            {
                // The limiter term is 10 times the k destruction term
                _k_prod(cell, point) = SmoothMath::min(
                    _k_prod(cell, point), 10.0 * _k_dest(cell, point), 0.0);
            }

            _k_source(cell, point)
                = (_k_prod(cell, point) - _k_dest(cell, point));

            // Turbulent dissipation rate terms
            _t_prod(cell, point)
                = _beta_0
                  - _gamma
                        * pow(_turb_specific_dissipation_rate(cell, point), 2.0)
                        * 2.0 * Sij_Sij;

            // Compute the destruction and cross diffusion term
            auto&& dkdxj_dwdxj = _t_cross(cell, point);
            dkdxj_dwdxj = 0.0;
            auto&& limit_source = _t_dest_nu(cell, point);
            limit_source = 0.0;
            for (int j = 0; j < num_space_dim; ++j)
            {
                limit_source += pow(
                    _grad_turb_specific_dissipation_rate(cell, point, j), 2.0);

                dkdxj_dwdxj
                    += _grad_turb_kinetic_energy(cell, point, j)
                       * _grad_turb_specific_dissipation_rate(cell, point, j);
            }

            _t_cross(cell, point)
                = _sigma_d * _turb_specific_dissipation_rate(cell, point)
                  * SmoothMath::min(dkdxj_dwdxj, 0.0, 0.0);

            _t_dest_nut(cell, point) = 2.0 * _turb_kinetic_energy(cell, point)
                                       * _sigma_t * limit_source;
            _t_dest_nu(cell, point)
                = 2.0 * _nu(cell, point) * limit_source
                  * SmoothMath::max(
                      1
                          / (SmoothMath::max(
                              _turb_specific_dissipation_rate(cell, point),
                              max_tol,
                              0.0)),
                      max_tol,
                      0.0);
            if (_limit_destruction
                && _t_dest_nu(cell, point) >= _t_dest_nut(cell, point))
            {
                _t_source(cell, point)
                    = _t_prod(cell, point)
                      - (SmoothMath::min(
                          4.0 / 3.0 * _beta_0,
                          _t_dest_nut(cell, point) + _t_dest_nu(cell, point),
                          0.0))
                      + _t_cross(cell, point);
            }
            else if (_limit_destruction
                     && _t_dest_nut(cell, point) > _t_dest_nu(cell, point))
            {
                _t_source(cell, point)
                    = _t_prod(cell, point)
                      - (SmoothMath::min(
                             4.0 / 3.0 * _beta_0, _t_dest_nu(cell, point), 0.0)
                         + _t_dest_nut(cell, point))
                      + _t_cross(cell, point);
            }
            else
            {
                _t_source(cell, point)
                    = _t_prod(cell, point)
                      - (_t_dest_nut(cell, point) + _t_dest_nu(cell, point))
                      + _t_cross(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_IMPL_HPP
