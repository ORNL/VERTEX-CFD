#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEENSTROPHY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEENSTROPHY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleEnstrophy<EvalType, Traits, NumSpaceDim>::IncompressibleEnstrophy(
    const panzer::IntegrationRule& ir)
    : _vorticity("vorticity", ir.dl_vector)
    , _enstrophy("enstrophy", ir.dl_scalar)
    , _divergence("divergence", ir.dl_scalar)
    , _mag_divergence("divergence_magnitude", ir.dl_scalar)
    , _nu("kinematic_viscosity", ir.dl_scalar)
{
    // Add evaluated fields
    this->addEvaluatedField(_vorticity);
    this->addEvaluatedField(_enstrophy);
    this->addEvaluatedField(_divergence);
    this->addEvaluatedField(_mag_divergence);

    // Add dependent field
    this->addDependentField(_nu);
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Incompressible Enstrophy " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleEnstrophy<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleEnstrophy<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _enstrophy.extent(1);
    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate vorticity vector
            if (num_space_dim < 3)
            {
                _vorticity(cell, point, 0)
                    = _grad_velocity[1](cell, point, 0)
                      - _grad_velocity[0](cell, point, 1);
                _vorticity(cell, point, 1) = 0.0;
            }
            else
            {
                _vorticity(cell, point, 0)
                    = _grad_velocity[2](cell, point, 1)
                      - _grad_velocity[1](cell, point, 2);
                _vorticity(cell, point, 1)
                    = _grad_velocity[0](cell, point, 2)
                      - _grad_velocity[2](cell, point, 0);
                _vorticity(cell, point, 2)
                    = _grad_velocity[1](cell, point, 0)
                      - _grad_velocity[0](cell, point, 1);
            }

            // Calculate enstrophy and divergence
            _enstrophy(cell, point) = 0.0;
            _divergence(cell, point) = 0.0;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _enstrophy(cell, point)
                    += pow(_vorticity(cell, point, dim), 2.0);
                _divergence(cell, point)
                    += _grad_velocity[dim](cell, point, dim);
            }

            _enstrophy(cell, point) *= _nu(cell, point);
            _mag_divergence(cell, point)
                = SmoothMath::abs(_divergence(cell, point), 0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEENSTROPHY_IMPL_HPP
