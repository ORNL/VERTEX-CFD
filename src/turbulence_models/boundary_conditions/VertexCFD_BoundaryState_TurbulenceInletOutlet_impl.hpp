#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// Inlet/outlet boundary condition for turbulence quantities
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
TurbulenceInletOutlet<EvalType, Traits, NumSpaceDim>::TurbulenceInletOutlet(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& bc_params,
    const std::string variable_name)
    : _boundary_variable("BOUNDARY_" + variable_name, ir.dl_scalar)
    , _boundary_grad_variable("BOUNDARY_GRAD_" + variable_name, ir.dl_vector)
    , _variable(variable_name, ir.dl_scalar)
    , _grad_variable("GRAD_" + variable_name, ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _inlet_value(bc_params.get<double>(variable_name + " Inlet Value"))
{
    // Add evaluated fields
    this->addEvaluatedField(_boundary_variable);
    this->addEvaluatedField(_boundary_grad_variable);

    // Add dependent fields
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    this->addDependentField(_variable);
    this->addDependentField(_grad_variable);
    this->addDependentField(_normals);

    this->setName(variable_name
                  + " Boundary State Turbulence Model Inlet/Outlet "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void TurbulenceInletOutlet<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
TurbulenceInletOutlet<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_variable.extent(1);
    const int num_grad_dim = _grad_variable.extent(2);
    const double smooth_ramp = 1.0e-8;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Compute \vec{vel} \cdot \vec{n}
            scalar_type vel_dot_n = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                vel_dot_n += _velocity[dim](cell, point)
                             * _normals(cell, point, dim);
            }

            // Ramping function for inlet/outlet
            const scalar_type outlet
                = SmoothMath::ramp(vel_dot_n, -smooth_ramp, smooth_ramp);

            // Assign boundary values
            _boundary_variable(cell, point) = (1.0 - outlet) * _inlet_value
                                              + outlet * _variable(cell, point);

            // Assign gradient
            for (int d = 0; d < num_grad_dim; ++d)
            {
                _boundary_grad_variable(cell, point, d)
                    = _grad_variable(cell, point, d);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_TURBULENCEINLETOUTLET_IMPL_HPP
