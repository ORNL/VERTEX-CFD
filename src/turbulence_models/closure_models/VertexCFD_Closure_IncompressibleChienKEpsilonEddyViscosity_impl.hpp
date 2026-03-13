#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleChienKEpsilonEddyViscosity<EvalType, Traits>::
    IncompressibleChienKEpsilonEddyViscosity(
        const panzer::IntegrationRule& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& turb_params)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
    , _nu("kinematic_viscosity", ir.dl_scalar)
    , _distance("distance", ir.dl_scalar)
    , _global_data(global_data)
    , _C_nu(0.09)
    , _C_tau(-0.0115)
    , _num_grad_dim(ir.spatial_dimension)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    _area = turb_params.get<double>("Boundary Surface Area");

    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_dissipation_rate);
    this->addDependentField(_nu);
    this->addDependentField(_distance);

    // Add evaluated fields
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("Chien K-Epsilon Incompressible Turbulent Eddy Viscosity "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleChienKEpsilonEddyViscosity<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    const auto& pl = *_global_data->pl;
    const std::string wall_shear_stress_name
        = "Friction Velocity - friction_velocity";
    _wall_shear_stress = pl.getValue<EvalType>(wall_shear_stress_name) / _area;

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleChienKEpsilonEddyViscosity<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const auto max_tol = 1.0e-12;
    using Kokkos::abs;
    using Kokkos::exp;
    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _nu_t(cell, point) = SmoothMath::max(
                _C_nu
                    * pow(SmoothMath::max(
                              _turb_kinetic_energy(cell, point), max_tol, 0.0),
                          2.0)
                    / SmoothMath::max(
                        _turb_dissipation_rate(cell, point), max_tol, 0.0)
                    * (1.0
                       - exp(_C_tau * _wall_shear_stress
                             * _distance(cell, point) / _nu(cell, point))),
                max_tol,
                0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECHIENKEPSILONEDDYVISCOSITY_IMPL_HPP
