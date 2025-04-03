#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_IMPL_HPP

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
IncompressibleSpalartAllmarasSource<EvalType, Traits, NumSpaceDim>::
    IncompressibleSpalartAllmarasSource(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _sa_var("spalart_allmaras_variable", ir.dl_scalar)
    , _distance("distance", ir.dl_scalar)
    , _grad_sa_var("GRAD_spalart_allmaras_variable", ir.dl_vector)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _sigma(2.0 / 3.0)
    , _kappa(0.41)
    , _c_b1(0.1355)
    , _c_b2(0.622)
    , _c_t3(1.2)
    , _c_t4(0.5)
    , _c_v1(7.1)
    , _c_v2(0.7)
    , _c_v3(0.9)
    , _c_w1(_c_b1 / (_kappa * _kappa) + (1.0 + _c_b2) / _sigma)
    , _c_w2(0.3)
    , _c_w3(2.0)
    , _rlim(10.0)
    , _sa_source("SOURCE_spalart_allmaras_equation", ir.dl_scalar)
    , _sa_prod("PRODUCTION_spalart_allmaras_equation", ir.dl_scalar)
    , _sa_dest("DESTRUCTION_spalart_allmaras_equation", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_sa_var);
    this->addDependentField(_distance);
    this->addDependentField(_grad_sa_var);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_sa_source);
    this->addEvaluatedField(_sa_prod);
    this->addEvaluatedField(_sa_dest);

    // Closure model name
    this->setName("Spalart-Allmaras Incompressible Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSpalartAllmarasSource<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleSpalartAllmarasSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _sa_var.extent(1);
    const double smooth_tol = 1.0e-8;
    using std::exp;
    using std::max;
    using std::pow;
    using std::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Cache SA variable and wall distance variable values
            const auto sa_var = _sa_var(cell, point);
            const auto d = _distance(cell, point);
            const scalar_type sa_ramp
                = SmoothMath::ramp(sa_var, -smooth_tol, smooth_tol);

            // Vorticity calculation
            scalar_type S2 = pow(_grad_velocity[0](cell, point, 1)
                                     - _grad_velocity[1](cell, point, 0),
                                 2.0);

            if (num_space_dim == 3)
            {
                S2 += pow(_grad_velocity[1](cell, point, 2)
                              - _grad_velocity[2](cell, point, 1),
                          2.0)
                      + pow(_grad_velocity[2](cell, point, 0)
                                - _grad_velocity[0](cell, point, 2),
                            2.0);
            }

            const scalar_type S = SmoothMath::max(sqrt(S2), smooth_tol, 0.0);

            // Functions for production term
            const scalar_type chi = sa_var / _nu;
            const scalar_type f_v1 = pow(chi, 3.0)
                                     / (pow(chi, 3.0) + pow(_c_v1, 3.0));
            const scalar_type f_v2 = 1.0 - chi / (1.0 + chi * f_v1);
            const scalar_type f_t2 = _c_t3 * exp(-_c_t4 * chi * chi);

            // Sbar
            const scalar_type Sbar = sa_var * f_v2 / (_kappa * _kappa * d * d);

            // Stilda
            scalar_type Stilda = S;

            if (Sbar > -_c_v2 * S)
            {
                Stilda += Sbar;
            }
            else
            {
                Stilda += S * (_c_v2 * _c_v2 * S + _c_v3 * Sbar)
                          / ((_c_v3 - 2 * _c_v2) * S - Sbar);
            }

            // Ensure Stilda is finite
            if (sqrt(pow(Stilda, 2.0)) < smooth_tol)
            {
                if (Stilda < 0)
                {
                    Stilda = -smooth_tol;
                }
                else
                {
                    Stilda = smooth_tol;
                }
            }

            // Production term calculation
            const scalar_type sa_prod_pos = _c_b1 * (1.0 - f_t2) * Stilda
                                            * sa_var;
            const scalar_type sa_prod_neg = _c_b1 * (1.0 - _c_t3) * S * sa_var;

            _sa_prod(cell, point) = sa_ramp * sa_prod_pos
                                    + (1.0 - sa_ramp) * sa_prod_neg;

            // Functions for destruction term
            const scalar_type r_sa = sa_var
                                     / (Stilda * _kappa * _kappa * d * d);
            const scalar_type r = SmoothMath::min(r_sa, _rlim, smooth_tol);
            const scalar_type g = r + _c_w2 * (pow(r, 6.0) - r);
            const scalar_type f_w = g
                                    * pow((1.0 + pow(_c_w3, 6.0))
                                              / (pow(g, 6.0) + pow(_c_w3, 6.0)),
                                          1.0 / 6.0);

            // Calculation of destruction term
            const scalar_type sa_dest_pos
                = -(_c_w1 * f_w - _c_b1 * f_t2 / (_kappa * _kappa))
                  * pow(sa_var / d, 2.0);
            const scalar_type sa_dest_neg = _c_w1 * pow(sa_var / d, 2.0);

            _sa_dest(cell, point) = sa_ramp * sa_dest_pos
                                    + (1.0 - sa_ramp) * sa_dest_neg;

            // Set SA source equal to sum of production and destruction
            _sa_source(cell, point) = _sa_prod(cell, point)
                                      + _sa_dest(cell, point);

            // Add SA gradient contribution
            for (int i = 0; i < num_space_dim; ++i)
            {
                _sa_source(cell, point)
                    += _c_b2 / _sigma * pow(_grad_sa_var(cell, point, i), 2.0);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASSOURCE_IMPL_HPP
