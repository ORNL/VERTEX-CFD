#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESHEARVARIABLES_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESHEARVARIABLES_IMPL_HPP

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
IncompressibleShearVariables<EvalType, Traits, NumSpaceDim>::
    IncompressibleShearVariables(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _tau_w("wall_shear_stress", ir.dl_scalar)
    , _u_tau("friction_velocity", ir.dl_scalar)
    , _normals("Side Normal", ir.dl_vector)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _rho(fluid_prop.constantDensity())
{
    // Add evaluated field
    this->addEvaluatedField(_tau_w);
    this->addEvaluatedField(_u_tau);

    // Add dependent field
    this->addDependentField(_normals);
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Incompressible Shear/Friction Variables "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleShearVariables<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleShearVariables<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _u_tau.extent(1);
    const int num_grad_dim = _normals.extent(2);
    using std::pow;
    using std::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Wall shear-stress
            _tau_w(cell, point) = 0.0;
            for (int i = 0; i < num_space_dim; ++i)
            {
                scalar_type sum = 0.0;
                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    sum += _normals(cell, point, dim)
                           * _grad_velocity[i](cell, point, dim);
                }
                _tau_w(cell, point) += pow(sum, 2.0);
            }
            _tau_w(cell, point)
                = sqrt(SmoothMath::max(_tau_w(cell, point), 1.0E-10, 0.0));
            _tau_w(cell, point) *= _nu;

            // Friction velocity
            _u_tau(cell, point) = sqrt(_tau_w(cell, point));
            _tau_w(cell, point) *= _rho;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLESHEARVARIABLES_IMPL_HPP
