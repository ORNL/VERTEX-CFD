#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONSOURCE_IMPL_HPP

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
IncompressibleChienKEpsilonSource<EvalType, Traits, NumSpaceDim>::
    IncompressibleChienKEpsilonSource(
        const panzer::IntegrationRule& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& user_params)
    : _nu("kinematic_viscosity", ir.dl_scalar)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
    , _distance("distance", ir.dl_scalar)
    , _global_data(global_data)
    , _C_nu(0.09)
    , _C_1(1.35)
    , _C_2(1.8)
    , _f_one(1.0)
    , _k_source("SOURCE_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_prod("PRODUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_dest("DESTRUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _e_source("SOURCE_turb_dissipation_rate_equation", ir.dl_scalar)
    , _e_prod("PRODUCTION_turb_dissipation_rate_equation", ir.dl_scalar)
    , _e_dest("DESTRUCTION_turb_dissipation_rate_equation", ir.dl_scalar)
{
    const Teuchos::ParameterList turb_list
        = user_params.sublist("Turbulence Parameters");

    // Get boundary surface area for wall shear stress
    _area = turb_list.get<double>("Boundary Surface Area");

    // Add dependent fields
    this->addDependentField(_nu);
    this->addDependentField(_nu_t);
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_dissipation_rate);
    this->addDependentField(_distance);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_k_source);
    this->addEvaluatedField(_k_prod);
    this->addEvaluatedField(_k_dest);
    this->addEvaluatedField(_e_source);
    this->addEvaluatedField(_e_prod);
    this->addEvaluatedField(_e_dest);

    // Closure model name
    this->setName("Chien K-Epsilon Incompressible Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleChienKEpsilonSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    const auto& pl = *_global_data->pl;
    const std::string wall_shear_stress_name
        = "Friction Velocity - friction_velocity";
    _wall_shear_stress = pl.getValue<EvalType>(wall_shear_stress_name) / _area;

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleChienKEpsilonSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-14;
    using Kokkos::exp;
    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Production term
            _k_prod(cell, point) = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    _k_prod(cell, point)
                        += pow(0.5
                                   * (_grad_velocity[i](cell, point, j)
                                      + _grad_velocity[j](cell, point, i)),
                               2.0);
                }
            }

            // Turbulent kinetic energy terms
            _k_prod(cell, point) *= (2.0 * _nu_t(cell, point));

            _k_dest(cell, point)
                = _turb_dissipation_rate(cell, point)
                  + 2 * _nu(cell, point) * _turb_kinetic_energy(cell, point)
                        / SmoothMath::max(
                            pow(_distance(cell, point), 2.0), max_tol, 0.0);

            _k_source(cell, point) = _k_prod(cell, point)
                                     - _k_dest(cell, point);

            // Turbulent dissipation rate terms
            _e_prod(cell, point)
                = _k_prod(cell, point) * _f_one * _C_1
                  * _turb_dissipation_rate(cell, point)
                  / SmoothMath::max(
                      _turb_kinetic_energy(cell, point), max_tol, 0.0);

            if (_turb_dissipation_rate(cell, point) > 0.0)
            {
                _e_dest(cell, point)
                    = _C_2
                          * (1.0
                             - 0.22
                                   * exp(-pow(
                                       pow(_turb_kinetic_energy(cell, point),
                                           2.0)
                                           / SmoothMath::max(
                                               6.0 * _nu(cell, point)
                                                   * _turb_dissipation_rate(
                                                       cell, point),
                                               max_tol,
                                               0.0),
                                       2.0)))
                          * pow(_turb_dissipation_rate(cell, point), 2.0)
                          / SmoothMath::max(
                              _turb_kinetic_energy(cell, point), max_tol, 0.0)
                      + 2.0 * _nu(cell, point)
                            * _turb_dissipation_rate(cell, point)
                            / SmoothMath::max(
                                pow(_distance(cell, point), 2.0), max_tol, 0.0)
                            * exp(-0.5 * _wall_shear_stress
                                  * _distance(cell, point) / _nu(cell, point));
            }
            else
            {
                _e_dest(cell, point) = 0.0;
            }

            _e_source(cell, point) = _e_prod(cell, point)
                                     - _e_dest(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONSOURCE_IMPL_HPP
