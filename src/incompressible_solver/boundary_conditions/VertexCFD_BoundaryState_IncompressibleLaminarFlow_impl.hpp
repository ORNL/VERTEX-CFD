#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELAMINARFLOW_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELAMINARFLOW_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::IncompressibleLaminarFlow(
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
    , _normals("Side Normal", ir.dl_vector)
    , _solve_temp(fluid_prop.solveTemperature())
    , _continuity_model_name(continuity_model_name)
    , _radius(bc_params.get<double>("Characteristic Radius"))
    , _vel_avg(bc_params.get<double>("Average velocity"))
    , _vel_max(num_space_dim == 2 ? 3.0 / 2.0 * _vel_avg : 2.0 * _vel_avg)
    , _T_bc(std::numeric_limits<double>::quiet_NaN())
{
    // Get origin coordinates if specified
    if (bc_params.isType<Teuchos::Array<double>>("Origin Coordinates"))
    {
        const auto origin_coord
            = bc_params.get<Teuchos::Array<double>>("Origin Coordinates");
        for (int dim = 0; dim < num_space_dim; ++dim)
            _origin_coord[dim] = origin_coord[dim];
    }
    else
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
            _origin_coord[dim] = 0.0;
    }

    if (continuity_model_name == "AC")
    {
        _continuity_model = ContinuityModel::AC;
    }
    else if (continuity_model_name == "EDAC")
    {
        _continuity_model = ContinuityModel::EDAC;
    }

    // Add evaluated fields
    this->addEvaluatedField(_boundary_lagrange_pressure);
    if (_continuity_model == ContinuityModel::EDAC)
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
    this->addDependentField(_normals);
    this->addDependentField(_lagrange_pressure);
    if (_continuity_model == ContinuityModel::EDAC)
        this->addDependentField(_grad_lagrange_pressure);
    if (_solve_temp)
        this->addDependentField(_grad_temperature);
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Boundary State Incompressible Laminar Flow "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    const int num_grad_dim = _boundary_grad_velocity[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::pow;
            // Set lagrange pressure
            _boundary_lagrange_pressure(cell, point)
                = _lagrange_pressure(cell, point);

            if (_solve_temp)
                _boundary_temperature(cell, point) = _T_bc;

            // Set velocity and gradients at boundaries.
            for (int d = 0; d < num_grad_dim; ++d)
            {
                if (_continuity_model == ContinuityModel::EDAC)
                {
                    _boundary_grad_lagrange_pressure(cell, point, d)
                        = _grad_lagrange_pressure(cell, point, d);
                }

                for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                {
                    // Calculate boundary velocity and gradient.
                    // Negative wall normal is used to show inward direction.
                    _boundary_velocity[vel_dim](cell, point) = 0.0;
                    for (int dim = 0; dim < num_space_dim; ++dim)
                    {
                        _boundary_velocity[vel_dim](cell, point)
                            -= _vel_max * _normals(cell, point, vel_dim)
                               * (1.0 / num_space_dim
                                  - pow(_ip_coords(cell, point, dim)
                                            - _origin_coord[dim],
                                        2)
                                        / (_radius * _radius));
                    }

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

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELAMINARFLOW_IMPL_HPP
