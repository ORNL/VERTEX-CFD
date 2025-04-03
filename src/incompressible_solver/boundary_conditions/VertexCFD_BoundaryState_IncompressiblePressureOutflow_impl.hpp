#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEPRESSUREOUTFLOW_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEPRESSUREOUTFLOW_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressiblePressureOutflow<EvalType, Traits, NumSpaceDim>::
    IncompressiblePressureOutflow(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& bc_params,
        const std::string& continuity_model_name)
    : _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _temperature("temperature", ir.dl_scalar)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _solve_temp(fluid_prop.solveTemperature())
    , _continuity_model_name(continuity_model_name)
    , _is_edac(continuity_model_name == "EDAC" ? true : false)
    , _p_back(bc_params.get<double>("Back Pressure"))
{
    // Add evaluated fields
    this->addEvaluatedField(_boundary_lagrange_pressure);
    if (_is_edac)
        this->addEvaluatedField(_boundary_grad_lagrange_pressure);
    if (_solve_temp)
    {
        this->addEvaluatedField(_boundary_temperature);
        this->addEvaluatedField(_boundary_grad_temperature);
    }
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _boundary_velocity, "BOUNDARY_velocity_");

    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_vector,
                                   _boundary_grad_velocity,
                                   "BOUNDARY_GRAD_velocity_");

    // Add dependent fields
    if (_is_edac)
        this->addDependentField(_grad_lagrange_pressure);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    if (_solve_temp)
    {
        this->addDependentField(_temperature);
        this->addDependentField(_grad_temperature);
    }

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Boundary State Incompressible Pressure Outflow "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressiblePressureOutflow<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressiblePressureOutflow<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_velocity[0].extent(1);
    const int num_grad_dim = _boundary_grad_velocity[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Assign velocity boundaries
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                _boundary_velocity[vel_dim](cell, point)
                    = _velocity[vel_dim](cell, point);
            }

            // Assign boundary conditions for primitive variables
            _boundary_lagrange_pressure(cell, point) = _p_back;

            // Temperature equation
            if (_solve_temp)
                _boundary_temperature(cell, point) = _temperature(cell, point);

            // Set boundary gradients
            for (int d = 0; d < num_grad_dim; ++d)
            {
                if (_is_edac)
                {
                    _boundary_grad_lagrange_pressure(cell, point, d)
                        = _grad_lagrange_pressure(cell, point, d);
                }

                if (_solve_temp)
                {
                    _boundary_grad_temperature(cell, point, d)
                        = _grad_temperature(cell, point, d);
                }

                for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                {
                    _boundary_grad_velocity[vel_dim](cell, point, d)
                        = _grad_velocity[vel_dim](cell, point, d);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEPRESSUREOUTFLOW_IMPL_HPP
