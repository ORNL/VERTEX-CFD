#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEROTATINGANNULUSEXACT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEROTATINGANNULUSEXACT_IMPL_HPP

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
IncompressibleRotatingAnnulusExact<EvalType, Traits, NumSpaceDim>::
    IncompressibleRotatingAnnulusExact(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop,
        const Teuchos::ParameterList& user_params)
    : _temperature("Exact_temperature", ir.dl_scalar)
    , _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _rho(fluid_prop.constantDensity())
    , _k(fluid_prop.constantThermalConductivity())
    , _ro(user_params.get<double>("Outer radius"))
    , _ri(user_params.get<double>("Inner radius"))
    , _kappa(_ri / _ro)
    , _omega(user_params.get<double>("Angular velocity"))
    , _To(user_params.get<double>("Outer wall temperature"))
    , _Ti(user_params.get<double>("Inner wall temperature"))
    , _ir_degree(ir.cubature_degree)
{
    using std::pow;

    // Evaluated fields
    this->addEvaluatedField(_temperature);
    this->addEvaluatedField(_lagrange_pressure);

    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");

    // Brinkman number
    _N = _nu * _rho * pow(_omega, 2.0) * pow(_ro, 2.0) / _k / (_To - _Ti)
         * pow(_kappa, 4.0) / pow(1.0 - pow(_kappa, 2.0), 2.0);

    this->setName("Exact Solution Rotating Annulus");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleRotatingAnnulusExact<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleRotatingAnnulusExact<EvalType, Traits, NumSpaceDim>::
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
IncompressibleRotatingAnnulusExact<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _temperature.extent(1);

    using std::log;
    using std::sqrt;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Note: an analytical solution is only available for the velocity
            // and temperature fields
            _lagrange_pressure(cell, point) = 0.0;

            // Get radius
            const double x = _ip_coords(cell, point, 0);
            const double y = _ip_coords(cell, point, 1);
            const double r = sqrt(x * x + y * y);
            const double ksi = r / _ro;

            // Calculate non-dimensional temperature
            const double theta = (1.0 - log(ksi) / log(_kappa))
                                 + _N
                                       * ((1 - 1 / (ksi * ksi))
                                          - (1 - 1 / (_kappa * _kappa))
                                                * log(ksi) / log(_kappa));

            // Back out exact temperature
            _temperature(cell, point) = theta * (_To - _Ti) + _Ti;

            // Tangential velocity
            const double u_phi = _omega * _ro * _ro / (_ro * _ro - _ri * _ri)
                                 * (r - (_ri * _ri / r));

            // Exact velocity
            _velocity[0](cell, point) = -u_phi * y / r;
            _velocity[1](cell, point) = u_phi * x / r;

            if (num_space_dim == 3)
                _velocity[2](cell, point) = 0.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEROTATINGANNULUSEXACT_IMPL_HPP
