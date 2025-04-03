#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleRealizableKEpsilonEddyViscosity<EvalType, Traits, NumSpaceDim>::
    IncompressibleRealizableKEpsilonEddyViscosity(
        const panzer::IntegrationRule& ir)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
    , _A_0(4.0)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_dissipation_rate);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName(
        "Realizable K-Epsilon Incompressible Turbulent Eddy Viscosity "
        + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleRealizableKEpsilonEddyViscosity<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleRealizableKEpsilonEddyViscosity<EvalType, Traits, NumSpaceDim>::
operator()(const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using std::acos;
    using std::cos;
    using std::pow;
    using std::sqrt;

    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const auto tol = 1.0e-10;
    const double sqrt_six = sqrt(6.0);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate mag square of symmetric and skew symmetric velocity
            // gradient components.
            // COMMENT: The Omega2 term must be modified if these equations
            // are to be solved in a rotating reference frame.
            scalar_type S2 = 0.0;
            scalar_type Omega2 = 0.0;
            scalar_type W = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    S2 += pow(0.5
                                  * (_grad_velocity[i](cell, point, j)
                                     + _grad_velocity[j](cell, point, i)),
                              2.0);
                    Omega2 += pow(0.5
                                      * (_grad_velocity[i](cell, point, j)
                                         - _grad_velocity[j](cell, point, i)),
                                  2.0);
                    for (int k = 0; k < num_space_dim; ++k)
                    {
                        W += 1.0 / 8.0
                             * (_grad_velocity[i](cell, point, j)
                                + _grad_velocity[j](cell, point, i))
                             * (_grad_velocity[j](cell, point, k)
                                + _grad_velocity[k](cell, point, j))
                             * (_grad_velocity[k](cell, point, i)
                                + _grad_velocity[i](cell, point, k));
                    }
                }
            }

            S2 = SmoothMath::max(S2, tol, 0.0);
            Omega2 = SmoothMath::max(Omega2, tol, 0.0);
            W /= pow(S2, 1.5);

            // Calculate parameters for viscosity
            const scalar_type Us = sqrt(S2 + Omega2);
            const scalar_type phi
                = 1.0 / 3.0
                  * acos(SmoothMath::max(
                      SmoothMath::min(sqrt_six * W, 1.0, 0.0), -1.0, 0.0));
            const scalar_type As = sqrt_six * cos(phi);
            const scalar_type C_nu
                = 1.0
                  / (_A_0
                     + (As * Us * _turb_kinetic_energy(cell, point)
                        / SmoothMath::max(
                            _turb_dissipation_rate(cell, point), tol, 0.0)));

            // Eddy viscosity calculation
            _nu_t(cell, point)
                = C_nu * pow(_turb_kinetic_energy(cell, point), 2.0)
                  / SmoothMath::max(
                      _turb_dissipation_rate(cell, point), tol, 0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONEDDYVISCOSITY_IMPL_HPP
