#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEPLANARPOISEUILLEEXACT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEPLANARPOISEUILLEEXACT_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include "Panzer_GlobalIndexer.hpp"
#include "Panzer_PureBasis.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressiblePlanarPoiseuilleExact<EvalType, Traits, NumSpaceDim>::
    IncompressiblePlanarPoiseuilleExact(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params)
    : _temperature("Exact_temperature", ir.dl_scalar)
    , _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _rho(fluid_prop.constantDensity())
    , _k(fluid_prop.constantThermalConductivity())
    , _cp(fluid_prop.constantHeatCapacity())
    , _ir_degree(ir.cubature_degree)
{
    // Read parameters from planar Poiseuille sublist
    const Teuchos::ParameterList pp_list
        = user_params.sublist("Planar Poiseuille Exact");

    _h_min = pp_list.get<double>("H min");
    _h_max = pp_list.get<double>("H max");
    _T_l = pp_list.get<double>("Lower wall temperature");
    _T_u = pp_list.get<double>("Upper wall temperature");
    _S_u = pp_list.get<double>("Momentum source");

    // Evaluated fields
    this->addEvaluatedField(_temperature);
    this->addEvaluatedField(_lagrange_pressure);

    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");

    this->setName("Exact Solution Planar Poiseuille");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressiblePlanarPoiseuilleExact<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressiblePlanarPoiseuilleExact<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressiblePlanarPoiseuilleExact<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _temperature.extent(1);

    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Note: an analytical solution is only available for the velocity
            // and temperature fields
            _lagrange_pressure(cell, point) = 0.0;

            // Get geometric and thermal parameters
            const double H = (_h_max - _h_min) / 2.0;
            const double y = _ip_coords(cell, point, 1);
            const double Pr = _cp * _rho * _nu / _k;
            const double dT = _T_l - _T_u;
            const double U = _S_u * H * H / 3.0 / _nu;
            const double E = U * U / _cp / dT;

            // Calculate excess temperature
            const double T_star = 0.5 * (1.0 - (y / H))
                                  + 3.0 / 4.0 * Pr * E
                                        * (1.0 - pow(y / H, 4.0));
            // Back out exact temperature
            _temperature(cell, point) = T_star * dT + _T_u;

            // Calculate exact velocity
            _velocity[0](cell, point) = 3.0 / 2.0 * U * (1.0 - pow(y / H, 2.0));
            _velocity[1](cell, point) = 0.0;

            if (num_space_dim == 3)
                _velocity[2](cell, point) = 0.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEPLANARPOISEUILLEEXACT_IMPL_HPP
