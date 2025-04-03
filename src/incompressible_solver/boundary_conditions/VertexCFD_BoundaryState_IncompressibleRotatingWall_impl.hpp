#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEROTATINGWALL_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEROTATINGWALL_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleRotatingWall<EvalType, Traits, NumSpaceDim>::IncompressibleRotatingWall(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const Teuchos::ParameterList& bc_params,
    const std::string& continuity_model_name)
    : _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _ir_degree(ir.cubature_degree)
    , _set_lagrange_pressure(bc_params.isType<double>("Lagrange Pressure"))
    , _lp_wall(std::numeric_limits<double>::quiet_NaN())
    , _solve_temp(fluid_prop.solveTemperature())
    , _continuity_model_name(continuity_model_name)
    , _is_edac(continuity_model_name == "EDAC" ? true : false)
    , _T_wall(std::numeric_limits<double>::quiet_NaN())
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
{
    // Compute the coefficients for the linear ramping in time f(x) = a * time
    // + b
    _time_init = bc_params.isType<double>("Time Initial")
                     ? bc_params.get<double>("Time Initial")
                     : 0.0;
    _time_final = bc_params.isType<double>("Time Final")
                      ? bc_params.get<double>("Time Final")
                      : 1.0E-06;

    if (_set_lagrange_pressure)
        _lp_wall = bc_params.get<double>("Lagrange Pressure");

    const auto angular_velocity_final
        = bc_params.get<double>("Angular Velocity");
    const auto angular_velocity_init = bc_params.isType<double>(
                                           "Angular Velocity "
                                           "Initial")
                                           ? bc_params.get<double>(
                                                 "Angular Velocity "
                                                 "Initial")
                                           : angular_velocity_final;
    const double dt = _time_final - _time_init;
    _a_vel = (angular_velocity_final - angular_velocity_init) / dt;
    _b_vel = angular_velocity_init - _a_vel * _time_init;

    // Add evaluated fields
    this->addEvaluatedField(_boundary_lagrange_pressure);
    if (_is_edac)
        this->addEvaluatedField(_boundary_grad_lagrange_pressure);

    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _boundary_velocity, "BOUNDARY_velocity_");

    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_vector,
                                   _boundary_grad_velocity,
                                   "BOUNDARY_GRAD_velocity_");
    if (_solve_temp)
    {
        _T_wall = bc_params.get<double>("Wall Temperature");
        this->addEvaluatedField(_boundary_temperature);
        this->addEvaluatedField(_boundary_grad_temperature);
    }

    // Add dependent fields
    this->addDependentField(_lagrange_pressure);
    if (_is_edac)
        this->addDependentField(_grad_lagrange_pressure);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    if (_solve_temp)
        this->addDependentField(_grad_temperature);

    this->setName("Boundary State Incompressible Rotating Wall "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleRotatingWall<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleRotatingWall<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Update time and make sure that 'time' only varies between '_time_init'
    // and '_time_final'
    const double time = workset.time < _time_init     ? _time_init
                        : workset.time >= _time_final ? _time_final
                                                      : workset.time;
    _angular_velocity = _a_vel * time + _b_vel;

    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleRotatingWall<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    const int num_grad_dim = _boundary_grad_velocity[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set lagrange pressure
            if (_set_lagrange_pressure)
                _boundary_lagrange_pressure(cell, point) = _lp_wall;

            else
                _boundary_lagrange_pressure(cell, point)
                    = _lagrange_pressure(cell, point);

            // Set wall temperature
            if (_solve_temp)
                _boundary_temperature(cell, point) = _T_wall;

            // Set boundary values for velocity components
            Kokkos::Array<double, num_space_dim> vel_bnd{};
            vel_bnd[0] = -_angular_velocity * _ip_coords(cell, point, 1);
            vel_bnd[1] = _angular_velocity * _ip_coords(cell, point, 0);
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                _boundary_velocity[vel_dim](cell, point) = vel_bnd[vel_dim];

            // Set gradients at boundaries.
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

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLEROTATINGWALL_IMPL_HPP
