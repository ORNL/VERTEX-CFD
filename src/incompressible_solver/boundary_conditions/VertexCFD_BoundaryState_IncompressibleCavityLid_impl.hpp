#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLECAVITYLID_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLECAVITYLID_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleCavityLid<EvalType, Traits, NumSpaceDim>::IncompressibleCavityLid(
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
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _solve_temp(fluid_prop.solveTemperature())
    , _continuity_model_name(continuity_model_name)
    , _is_edac(continuity_model_name == "EDAC" ? true : false)
    , _wall_dir(bc_params.get<int>("Wall Normal Direction"))
    , _vel_dir(bc_params.get<int>("Velocity Direction"))
    , _h(bc_params.get<double>("Half Width"))
    , _u_wall(bc_params.get<double>("Wall Velocity"))
    , _T_bc(std::numeric_limits<double>::quiet_NaN())
{
    // Check that dimensions make sense
    if (_wall_dir >= num_space_dim)
    {
        const std::string msg
            = "Wall normal direction greater than "
              "number of solution dimensions in Cavity Lid boundary "
              "condition.";

        throw std::runtime_error(msg);
    }
    if (_vel_dir >= num_space_dim)
    {
        const std::string msg
            = "Velocity direction greater than "
              "number of solution dimensions in Cavity Lid boundary "
              "condition.";

        throw std::runtime_error(msg);
    }
    if (_wall_dir == _vel_dir)
    {
        const std::string msg
            = "Velocity direction is same as wall normal in "
              "Cavity Lid boundary condition.";

        throw std::runtime_error(msg);
    }

    // Add evaluated fields
    this->addEvaluatedField(_boundary_lagrange_pressure);
    if (_is_edac)
        this->addEvaluatedField(_boundary_grad_lagrange_pressure);
    if (_solve_temp)
    {
        _T_bc = bc_params.get<double>("Temperature");
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
    this->addDependentField(_lagrange_pressure);
    if (_is_edac)
        this->addDependentField(_grad_lagrange_pressure);
    if (_solve_temp)
        this->addDependentField(_grad_temperature);
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Boundary State Incompressible Cavity Lid "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleCavityLid<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleCavityLid<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleCavityLid<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    const int num_grad_dim = _boundary_grad_velocity[0].extent(2);

    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set lagrange pressure
            _boundary_lagrange_pressure(cell, point)
                = _lagrange_pressure(cell, point);

            // Set boundary values for velocity components
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                if (vel_dim == _vel_dir)
                {
                    _boundary_velocity[vel_dim](cell, point) = _u_wall;

                    for (int dim = 0; dim < num_space_dim; ++dim)
                    {
                        if (dim != _wall_dir)
                        {
                            _boundary_velocity[vel_dim](cell, point) *= pow(
                                1.0
                                    - pow(_ip_coords(cell, point, dim) / _h,
                                          18.0),
                                2.0);
                        }
                    }
                }
                else
                {
                    _boundary_velocity[vel_dim](cell, point) = 0.0;
                }
            }

            if (_solve_temp)
                _boundary_temperature(cell, point) = _T_bc;

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

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLECAVITYLID_IMPL_HPP
