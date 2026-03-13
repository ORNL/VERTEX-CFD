#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLENOSLIP_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLENOSLIP_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleNoSlip<EvalType, Traits, NumSpaceDim>::IncompressibleNoSlip(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& fluid_params,
    const Teuchos::ParameterList& bc_params,
    const bool is_edac)
    : _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_temperature("BOUNDARY_temperature", ir.dl_scalar)
    , _boundary_grad_temperature("BOUNDARY_GRAD_temperature", ir.dl_vector)
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _temperature("temperature", ir.dl_scalar)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _kappa("thermal_conductivity", ir.dl_scalar)
    , _solve_temp(fluid_params.get<bool>("Build Temperature Equation"))
    , _set_lagrange_pressure(bc_params.isType<double>("Lagrange Pressure"))
    , _lp_wall(std::numeric_limits<double>::quiet_NaN())
    , _is_edac(is_edac)
    , _T_wall(std::numeric_limits<double>::quiet_NaN())
    , _wall_flux(std::numeric_limits<double>::quiet_NaN())
    , _temperature_profile(TempProfile::isothermal)
{
    // Get temperature profile
    if (bc_params.isType<std::string>("Temperature Profile"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<TempProfile>(
                Teuchos::tuple<std::string>("Isothermal", "Adiabatic", "Flux"),
                "Isothermal"));
        _temperature_profile = type_validator->getIntegralValue(
            bc_params.get<std::string>("Temperature Profile"));
    }

    if (_set_lagrange_pressure)
        _lp_wall = bc_params.get<double>("Lagrange Pressure");

    // Evaluated fields
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
        this->addEvaluatedField(_boundary_temperature);
        this->addEvaluatedField(_boundary_grad_temperature);
    }

    // Dependent fields
    this->addDependentField(_lagrange_pressure);
    if (_is_edac)
    {
        this->addDependentField(_grad_lagrange_pressure);
        this->addDependentField(_normals);
    }

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    if (_solve_temp)
    {
        this->addDependentField(_grad_temperature);
        if (_temperature_profile == TempProfile::isothermal)
        {
            _T_wall = bc_params.get<double>("Wall Temperature");
        }
        else if (_temperature_profile == TempProfile::adiabatic)
        {
            this->addDependentField(_temperature);
            if (!_is_edac)
                this->addDependentField(_normals);
        }
        else if (_temperature_profile == TempProfile::flux)
        {
            this->addDependentField(_temperature);
            this->addDependentField(_kappa);
            if (!_is_edac)
                this->addDependentField(_normals);
            _wall_flux = bc_params.get<double>("Wall Flux");
        }
    }

    this->setName("Boundary State Incompressible No-Slip");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleNoSlip<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleNoSlip<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_lagrange_pressure.extent(1);
    const int num_grad_dim = _boundary_grad_velocity[0].extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set lagrange pressure
            if (_set_lagrange_pressure)
            {
                _boundary_lagrange_pressure(cell, point) = _lp_wall;
            }
            else
            {
                _boundary_lagrange_pressure(cell, point)
                    = _lagrange_pressure(cell, point);
            }

            // Set boundary values
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                _boundary_velocity[vel_dim](cell, point) = 0.0;

            if (_solve_temp)
            {
                if (_temperature_profile == TempProfile::isothermal)
                {
                    _boundary_temperature(cell, point) = _T_wall;
                }
                else
                {
                    _boundary_temperature(cell, point)
                        = _temperature(cell, point);
                }
            }

            // Set gradients at the boundaries.
            for (int d = 0; d < num_grad_dim; ++d)
            {
                if (_is_edac)
                {
                    // Set lagrange pressure gradient
                    _boundary_grad_lagrange_pressure(cell, point, d)
                        = _grad_lagrange_pressure(cell, point, d);

                    for (int grad_dim = 0; grad_dim < num_space_dim; ++grad_dim)
                    {
                        _boundary_grad_lagrange_pressure(cell, point, d)
                            -= (_grad_lagrange_pressure(cell, point, grad_dim)
                                * _normals(cell, point, grad_dim))
                               * _normals(cell, point, d);
                    }
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

                    if (_temperature_profile != TempProfile::isothermal)
                    {
                        for (int grad_dim = 0; grad_dim < num_space_dim;
                             ++grad_dim)
                        {
                            _boundary_grad_temperature(cell, point, d)
                                -= (_grad_temperature(cell, point, grad_dim)
                                    * _normals(cell, point, grad_dim))
                                   * _normals(cell, point, d);
                        }
                        if (_temperature_profile == TempProfile::flux)
                        {
                            _boundary_grad_temperature(cell, point, d)
                                += (_wall_flux / _kappa(cell, point))
                                   * _normals(cell, point, d);
                        }
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLENOSLIP_IMPL_HPP
