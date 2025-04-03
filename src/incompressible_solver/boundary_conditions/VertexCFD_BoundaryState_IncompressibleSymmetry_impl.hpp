#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLESYMMETRY_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLESYMMETRY_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleSymmetry<EvalType, Traits, NumSpaceDim>::IncompressibleSymmetry(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const std::string& continuity_model_name)
    : _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _temperature("temperature", ir.dl_scalar)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _solve_temp(fluid_prop.solveTemperature())
    , _continuity_model_name(continuity_model_name)
    , _is_edac(continuity_model_name == "EDAC" ? true : false)
{
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
        this->addDependentField(_temperature);
        this->addEvaluatedField(_boundary_temperature);
        this->addDependentField(_grad_temperature);
        this->addEvaluatedField(_boundary_grad_temperature);
    }

    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    this->addDependentField(_lagrange_pressure);
    if (_is_edac)
        this->addDependentField(_grad_lagrange_pressure);

    this->addDependentField(_normals);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Boundary State Incompressible Free Slip "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleSymmetry<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleSymmetry<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    const int num_grad_dim = _normals.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set boundary Lagrange pressure
            _boundary_lagrange_pressure(cell, point)
                = _lagrange_pressure(cell, point);

            // Compute components of flow variables normal to surface
            scalar_type vel_dot_n = 0.0;
            scalar_type grad_T_dot_n = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                vel_dot_n += _velocity[dim](cell, point)
                             * _normals(cell, point, dim);

                if (_solve_temp)
                {
                    grad_T_dot_n += _grad_temperature(cell, point, dim)
                                    * _normals(cell, point, dim);
                }
            }

            Kokkos::Array<scalar_type, num_space_dim> grad_vel_dot_n{};
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                grad_vel_dot_n[vel_dim] = 0.0;

                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    grad_vel_dot_n[vel_dim]
                        += _grad_velocity[vel_dim](cell, point, dim)
                           * _normals(cell, point, dim);
                }
            }

            if (_solve_temp)
                _boundary_temperature(cell, point) = _temperature(cell, point);

            // Set boundary velocity and boundary gradients
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                if (_is_edac)
                {
                    _boundary_grad_lagrange_pressure(cell, point, dim)
                        = _grad_lagrange_pressure(cell, point, dim);
                }

                _boundary_velocity[dim](cell, point)
                    = _velocity[dim](cell, point)
                      - vel_dot_n * _normals(cell, point, dim);

                for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                {
                    _boundary_grad_velocity[vel_dim](cell, point, dim)
                        = _grad_velocity[vel_dim](cell, point, dim)
                          - grad_vel_dot_n[vel_dim]
                                * _normals(cell, point, dim);
                }

                if (_solve_temp)
                {
                    _boundary_grad_temperature(cell, point, dim)
                        = _grad_temperature(cell, point, dim)
                          - grad_T_dot_n * _normals(cell, point, dim);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLESYMMETRY_IMPL_HPP
