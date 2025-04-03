#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEDIRICHLET_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEDIRICHLET_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// This function should be used for Dirichlet boundary conditions. A ramping
// in time can be enabled for all variables or only a few. Be aware that the
// ramping in time does not include any logic with the characteristics and thus
// should only be used for non-transient runs.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleDirichlet<EvalType, Traits, NumSpaceDim>::IncompressibleDirichlet(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const Teuchos::ParameterList& bc_params,
    const std::string& continuity_model_name)
    : _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _solve_temp(fluid_prop.solveTemperature())
    , _continuity_model_name(continuity_model_name)
    , _is_edac(continuity_model_name == "EDAC" ? true : false)
{
    // Calculate the coefficients 'a' and 'b' for the linear time ramping
    // f(t) = a * t + b for each variable
    _time_init = bc_params.isType<double>("Time Initial")
                     ? bc_params.get<double>(
                           "Time "
                           "Initial")
                     : 0.0;
    _time_final = bc_params.isType<double>("Time Final")
                      ? bc_params.get<double>("Time Final")
                      : 1.0E-06;
    const double dt = _time_final - _time_init;

    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        const std::string vel_string = "velocity_" + std::to_string(vel_dim);
        const auto vel_final = bc_params.get<double>(vel_string);
        const auto vel_init = bc_params.isType<double>(vel_string + "_init")
                                  ? bc_params.get<double>(vel_string + "_init")
                                  : vel_final;
        _a_vel[vel_dim] = (vel_final - vel_init) / dt;
        _b_vel[vel_dim] = vel_init - _a_vel[vel_dim] * _time_init;
    }

    // Add evaluated fields
    this->addEvaluatedField(_boundary_lagrange_pressure);
    if (_is_edac)
        this->addEvaluatedField(_boundary_grad_lagrange_pressure);
    if (_solve_temp)
    {
        _T_dirichlet = bc_params.get<double>("temperature");
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
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");
    this->addDependentField(_lagrange_pressure);
    if (_is_edac)
        this->addDependentField(_grad_lagrange_pressure);
    if (_solve_temp)
        this->addDependentField(_grad_temperature);

    this->setName("Boundary State Incompressible Dirichlet "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleDirichlet<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Get time and make sure it only varies between '_time_init' and
    // '_time_final'
    _time = std::max(workset.time, _time_init);
    _time = std::min(_time, _time_final);

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleDirichlet<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    const int num_grad_dim = _boundary_grad_velocity[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Assign time-dependent boundary values
            _boundary_lagrange_pressure(cell, point)
                = _lagrange_pressure(cell, point);
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                _boundary_velocity[vel_dim](cell, point)
                    = _a_vel[vel_dim] * _time + _b_vel[vel_dim];
            }
            if (_solve_temp)
                _boundary_temperature(cell, point) = _T_dirichlet;

            // Set gradients
            for (int d = 0; d < num_grad_dim; ++d)
            {
                if (_is_edac)
                {
                    _boundary_grad_lagrange_pressure(cell, point, d)
                        = _grad_lagrange_pressure(cell, point, d);
                }

                for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                {
                    _boundary_grad_velocity[vel_dim](cell, point, d)
                        = _grad_velocity[vel_dim](cell, point, d);
                }

                if (_solve_temp)
                {
                    _boundary_grad_temperature(cell, point, d)
                        = _grad_temperature(cell, point, d);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEDIRICHLET_IMPL_HPP
