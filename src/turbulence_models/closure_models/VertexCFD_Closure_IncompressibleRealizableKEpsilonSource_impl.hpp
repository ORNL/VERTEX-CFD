#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONSOURCE_IMPL_HPP

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
IncompressibleRealizableKEpsilonSource<EvalType, Traits, NumSpaceDim>::
    IncompressibleRealizableKEpsilonSource(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    , _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _C_2(1.9)
    , _k_source("SOURCE_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_prod("PRODUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _k_dest("DESTRUCTION_turb_kinetic_energy_equation", ir.dl_scalar)
    , _e_source("SOURCE_turb_dissipation_rate_equation", ir.dl_scalar)
    , _e_prod("PRODUCTION_turb_dissipation_rate_equation", ir.dl_scalar)
    , _e_dest("DESTRUCTION_turb_dissipation_rate_equation", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_nu_t);
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_dissipation_rate);

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
    this->setName("Realizable K-Epsilon Incompressible Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleRealizableKEpsilonSource<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleRealizableKEpsilonSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-10;
    using std::pow;
    using std::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate S and mag square of velocity gradient
            scalar_type S2 = 0.0;
            scalar_type grad_u_sqr = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    S2 += pow(0.5
                                  * (_grad_velocity[i](cell, point, j)
                                     + _grad_velocity[j](cell, point, i)),
                              2.0);
                    grad_u_sqr += pow(_grad_velocity[i](cell, point, j), 2.0);
                }
            }

            const scalar_type S = sqrt(SmoothMath::max(2.0 * S2, max_tol, 0.0));

            // Calculate model coefficients
            const scalar_type eta
                = S * _turb_kinetic_energy(cell, point)
                  / SmoothMath::max(
                      _turb_dissipation_rate(cell, point), max_tol, 0.0);
            const scalar_type C_1
                = SmoothMath::max(0.43, eta / (5.0 + eta), 0.0);

            // Turbulent kinetic energy terms (same as standard K-Epsilon)
            _k_prod(cell, point) = _nu_t(cell, point) * grad_u_sqr;
            _k_dest(cell, point) = -_turb_dissipation_rate(cell, point);
            _k_source(cell, point) = _k_prod(cell, point)
                                     + _k_dest(cell, point);

            // Turbulent dissipation rate terms (modified for RKE model)
            _e_prod(cell, point) = C_1 * S
                                   * _turb_dissipation_rate(cell, point);
            _e_dest(cell, point)
                = -_C_2 * pow(_turb_dissipation_rate(cell, point), 2.0)
                  / SmoothMath::max(
                      _turb_kinetic_energy(cell, point)
                          + sqrt(SmoothMath::max(
                              _nu * _turb_dissipation_rate(cell, point),
                              max_tol,
                              0.0)),
                      max_tol,
                      0.0);
            _e_source(cell, point) = _e_prod(cell, point)
                                     + _e_dest(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONSOURCE_IMPL_HPP
