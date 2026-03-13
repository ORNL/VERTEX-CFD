#ifndef VERTEXCFD_CLOSURE_CONSTANTSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_CONSTANTSOURCE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleConstantSource<EvalType, Traits, NumSpaceDim>::
    IncompressibleConstantSource(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& fluid_params,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& closure_params)
    : _energy_source("CONSTANT_SOURCE_energy", ir.dl_scalar)
    , _global_data(global_data)
    , _solve_temp(fluid_params.get<bool>("Build Temperature Equation"))
    , _solve_ind_less_mhd(fluid_params.get<bool>("Build Inductionless MHD "
                                                 "Equation"))
    , _flow_direction(FlowDirection::x)
    , _const_vol_flow(false)
{
    if (closure_params.isType<bool>("Constant Volumetric Flow Rate"))
        _const_vol_flow
            = closure_params.get<bool>("Constant Volumetric Flow Rate");

    if (_const_vol_flow)
    {
        _m_target = closure_params.get<double>("Target Volumetric Flow Rate");
        _bottom_wall_area
            = closure_params.get<double>("Bottom Wall Surface Area");
        _inlet_wall_area
            = closure_params.get<double>("Inlet Wall Surface Area");
        _flow_direction_idx = 0;
        if (closure_params.isType<std::string>("Flow Direction"))
        {
            const auto type_validator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<FlowDirection>(
                    Teuchos::tuple<std::string>("x", "y", "z"), "x"));
            _flow_direction = type_validator->getIntegralValue(
                closure_params.get<std::string>("Flow Direction"));
            if (_flow_direction == FlowDirection::x)
                _flow_direction_idx = 0;
            if (_flow_direction == FlowDirection::y)
                _flow_direction_idx = 1;
            if (_flow_direction == FlowDirection::z)
                _flow_direction_idx = 2;
        }
    }
    else
    {
        const auto mom_input_source
            = closure_params.get<Teuchos::Array<double>>("Momentum Source");
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            _mom_input_source[dim] = mom_input_source[dim];
        }
    }

    // Evaluated fields
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _momentum_source,
                                   "CONSTANT_SOURCE_"
                                   "momentum_");

    if (_solve_temp)
    {
        _energy_input_source = closure_params.get<double>("Energy Source");
        this->addEvaluatedField(_energy_source);
    }

    // Dependent fields
    this->setName("Incompressible Constant Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void
    IncompressibleConstantSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
        typename Traits::EvalData /* workset */)
{
    if (_const_vol_flow)
    {
        const auto& pl = *_global_data->pl;
        const std::string m_bc_name = "Volumetric Flow Rate - velocity_"
                                      + std::to_string(_flow_direction_idx);
        _m_bc = pl.getValue<EvalType>(m_bc_name);

        const std::string wall_shear_stress_name
            = "Wall Shear Stress - wall_shear_stress";
        _wall_shear_stress = pl.getValue<EvalType>(wall_shear_stress_name)
                             / _bottom_wall_area;

        if (_solve_ind_less_mhd)
        {
            const std::string lorentz_name
                = "Lorentz Force - VOLUMETRIC_SOURCE_momentum_"
                  + std::to_string(_flow_direction_idx);
            _lorentz_force = pl.getValue<EvalType>(lorentz_name)
                             / _inlet_wall_area;
        }
    }
    for (int mom_dim = 0; mom_dim < num_space_dim; ++mom_dim)
    {
        if (_const_vol_flow)
        {
            if (mom_dim == _flow_direction_idx)
            {
                // Calculate the friction loss (wall shear stress on the bottom
                // and top surfaces (bottom = top))
                _mom_input_source[mom_dim]
                    = _m_target / _m_bc
                      * (2.0 * _wall_shear_stress / _inlet_wall_area);
                // If lorentz force is available, add the contribution from the
                // magnetic field induced drag
                if (_solve_ind_less_mhd)
                    _mom_input_source[mom_dim] -= _m_target / _m_bc
                                                  * _lorentz_force;
            }
            else
                _mom_input_source[mom_dim] = 0.0;
        }

        Kokkos::deep_copy(_momentum_source[mom_dim].get_view(),
                          _mom_input_source[mom_dim]);
    }

    if (_solve_temp)
        Kokkos::deep_copy(_energy_source.get_view(), _energy_input_source);
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONSTANTSOURCE_IMPL_HPP
