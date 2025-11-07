#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <algorithm>
#include <math.h>
#include <type_traits>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleSSTSource<EvalType, Traits, NumSpaceDim>::IncompressibleSSTSource(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& user_params)
    : _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _grad_turb_kinetic_energy("GRAD_turb_kinetic_energy", ir.dl_vector)
    , _grad_turb_specific_dissipation_rate(
          "GRAD_turb_specific_dissipation_rate", ir.dl_vector)
    , _sst_blending_function("sst_blending_function", ir.dl_scalar)
    , _beta_star(0.09)
    , _kappa(0.41)
    , _beta_1(0.075)
    , _beta_2(0.0828)
    , _sigma_w1(0.5)
    , _sigma_w2(0.856)
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

        if (turb_list.isSublist("SST K-Omega Parameters"))
        {
            Teuchos::ParameterList sst_list
                = turb_list.sublist("SST K-Omega Parameters");

            if (sst_list.isType<double>("beta_star"))
            {
                _beta_star = sst_list.get<double>("beta_star");
            }

            if (sst_list.isType<double>("kappa"))
            {
                _kappa = sst_list.get<double>("kappa");
            }

            if (sst_list.isType<double>("beta_1"))
            {
                _beta_1 = sst_list.get<double>("beta_1");
            }

            if (sst_list.isType<double>("beta_2"))
            {
                _beta_2 = sst_list.get<double>("beta_2");
            }

            if (sst_list.isType<double>("sigma_w1"))
            {
                _sigma_w1 = sst_list.get<double>("sigma_w1");
            }

            if (sst_list.isType<double>("sigma_w2"))
            {
                _sigma_w2 = sst_list.get<double>("sigma_w2");
            }

            if (sst_list.isType<bool>("Limit Production Term"))
            {
                _limit_production
                    = sst_list.get<bool>("Limit Production Term");
            }
        }

        _gamma_1 = _beta_1 / _beta_star
                   - _sigma_w1 * _kappa * _kappa / std::sqrt(_beta_star);
        _gamma_2 = _beta_2 / _beta_star
                   - _sigma_w2 * _kappa * _kappa / std::sqrt(_beta_star);
    }
    // Add dependent fields
    this->addDependentField(_nu_t);
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);
    this->addDependentField(_grad_turb_kinetic_energy);
    this->addDependentField(_grad_turb_specific_dissipation_rate);
    this->addDependentField(_sst_blending_function);

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
    this->setName("SST Incompressible Source " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSSTSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleSSTSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-10;
    using Kokkos::abs;
    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& Sij_Sij = _k_source(cell, point);
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

            auto&& dkdxj_domegadxj = _w_source(cell, point);
            dkdxj_domegadxj = 0.0;
            for (int j = 0; j < num_space_dim; ++j)
            {
                dkdxj_domegadxj
                    += _grad_turb_kinetic_energy(cell, point, j)
                       * _grad_turb_specific_dissipation_rate(cell, point, j);
            }

            // Turbulent kinetic energy terms, these are the same as standard
            // k-omega model, in the standard model the unlimited value is used
            // in other equations
            auto&& pk_unlimited = _w_dest(cell, point);
            pk_unlimited = _k_prod(cell, point) = 2.0 * _nu_t(cell, point)
                                                  * Sij_Sij;
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

            // Turbulent dissipation rate terms, destruction is the same as
            // standard k-omega model but with a different coefficient
            _w_prod(cell, point)
                = (_gamma_1 * _sst_blending_function(cell, point)
                   + _gamma_2 * (1.0 - _sst_blending_function(cell, point)))
                  * pk_unlimited
                  / SmoothMath::max(_nu_t(cell, point), max_tol, 0.0);
            _w_dest(cell, point)
                = -(_beta_1 * _sst_blending_function(cell, point)
                    + _beta_2 * (1.0 - _sst_blending_function(cell, point)))
                  * pow(_turb_specific_dissipation_rate(cell, point), 2);
            // Compute the cross diffusion term
            _w_cross(cell, point)
                = 2.0 * (1.0 - _sst_blending_function(cell, point)) * _sigma_w2
                  * dkdxj_domegadxj
                  / SmoothMath::max(
                      _turb_specific_dissipation_rate(cell, point),
                      max_tol,
                      0.0);
            _w_source(cell, point) = _w_prod(cell, point) + _w_dest(cell, point)
                                     + _w_cross(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTSOURCE_IMPL_HPP
