#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_IMPL_HPP

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
IncompressibleKOmegaSource<EvalType, Traits, NumSpaceDim>::IncompressibleKOmegaSource(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& user_params)
    : _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    ,

    _grad_turb_kinetic_energy("GRAD_turb_kinetic_energy", ir.dl_vector)
    , _grad_turb_specific_dissipation_rate(
          "GRAD_turb_specific_dissipation_rate", ir.dl_vector)
    , _beta_star(0.09)
    , _gamma(0.52)
    , _beta_0(0.0708)
    , _sigma_d(0.125)
    , _limit_production(true)
    , _k_source("SOURCE_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_prod("PRODUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_dest("DESTRUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _w_source("SOURCE_turb_specific_dissipation_rate_equation", ir.dl_scalar)
    , _w_prod("PRODUCTION_turb_specific_dissipation_rate_equation",
              ir.dl_scalar)
    , _w_dest("DESTRUCTION_turb_specific_dissipation_rate_equation",
              ir.dl_scalar)
    , _w_cross("CROSS_DIFFUSION_turb_specific_dissipation_rate_equation",
               ir.dl_scalar)
{
    // Check for user-defined coefficients or parameters
    if (user_params.isSublist("Turbulence Parameters"))
    {
        Teuchos::ParameterList turb_list
            = user_params.sublist("Turbulence Parameters");

        if (turb_list.isType<double>("beta_star"))
        {
            _beta_star = turb_list.get<double>("beta_star");
        }

        if (turb_list.isType<double>("gamma"))
        {
            _gamma = turb_list.get<double>("gamma");
        }

        if (turb_list.isType<double>("beta_0"))
        {
            _beta_0 = turb_list.get<double>("beta_0");
        }

        if (turb_list.isType<double>("sigma_d"))
        {
            _sigma_d = turb_list.get<double>("sigma_d");
        }

        if (turb_list.isType<bool>("Limit Production Term"))
        {
            _limit_production = turb_list.get<bool>("Limit Production Term");
        }
    }

    // Add dependent fields
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
    this->addEvaluatedField(_w_source);
    this->addEvaluatedField(_w_prod);
    this->addEvaluatedField(_w_dest);
    this->addEvaluatedField(_w_cross);

    // Closure model name
    this->setName("K-Omega Incompressible Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleKOmegaSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleKOmegaSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-10;
    using std::abs;
    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            scalar_type Sij_Sij = 0.0;
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

            scalar_type chi_w = 0.0;
            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    for (int k = 0; k < num_space_dim; ++k)
                    {
                        chi_w += 0.125
                                 * (_grad_velocity[i](cell, point, j)
                                    - _grad_velocity[j](cell, point, i))
                                 * (_grad_velocity[j](cell, point, k)
                                    - _grad_velocity[k](cell, point, j))
                                 * (_grad_velocity[i](cell, point, k)
                                    + _grad_velocity[k](cell, point, i));
                    }
                }
            }

            chi_w = abs(
                chi_w
                / SmoothMath::max(
                    pow(_beta_star
                            * _turb_specific_dissipation_rate(cell, point),
                        3.0),
                    max_tol,
                    0.0));

            const scalar_type f_beta = (1.0 + 85.0 * chi_w)
                                       / (1.0 + 100.0 * chi_w);

            // Turbulent kinetic energy terms
            _k_prod(cell, point) = 2.0 * _nu_t(cell, point) * Sij_Sij;

            _k_dest(cell, point)
                = -_beta_star * _turb_specific_dissipation_rate(cell, point)
                  * _turb_kinetic_energy(cell, point);

            if (_limit_production)
            {
                // The limiter term is -20 times the k destruction term
                _k_prod(cell, point) = SmoothMath::min(
                    _k_prod(cell, point), -20.0 * _k_dest(cell, point), 0.0);
            }

            _k_source(cell, point) = _k_prod(cell, point)
                                     + _k_dest(cell, point);

            // Turbulent dissipation rate terms
            _w_prod(cell, point)
                = _gamma * _turb_specific_dissipation_rate(cell, point) * 2.0
                  * _nu_t(cell, point) * Sij_Sij
                  / SmoothMath::max(
                      _turb_kinetic_energy(cell, point), max_tol, 0.0);
            _w_dest(cell, point)
                = -_beta_0 * f_beta
                  * pow(_turb_specific_dissipation_rate(cell, point), 2);

            // Compute the cross diffusion term
            _w_cross(cell, point) = 0.0;
            scalar_type dkdxj_dwdxj = 0.0;
            for (int j = 0; j < num_space_dim; ++j)
            {
                dkdxj_dwdxj
                    += _grad_turb_kinetic_energy(cell, point, j)
                       * _grad_turb_specific_dissipation_rate(cell, point, j);
            }
            if (dkdxj_dwdxj > 0.0)
            {
                _w_cross(cell, point)
                    = _sigma_d * dkdxj_dwdxj
                      / SmoothMath::max(
                          _turb_specific_dissipation_rate(cell, point),
                          max_tol,
                          0.0);
            }

            _w_source(cell, point) = _w_prod(cell, point) + _w_dest(cell, point)
                                     + _w_cross(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_IMPL_HPP
