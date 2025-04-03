#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleKEpsilonEddyViscosity<EvalType, Traits>::
    IncompressibleKEpsilonEddyViscosity(const panzer::IntegrationRule& ir)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
    , _C_nu(0.09)
    , _num_grad_dim(ir.spatial_dimension)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_dissipation_rate);

    // Add evaluated fields
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("K-Epsilon Incompressible Turbulent Eddy Viscosity "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleKEpsilonEddyViscosity<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleKEpsilonEddyViscosity<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const auto max_tol = 1.0e-10;
    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _nu_t(cell, point)
                = _C_nu * pow(_turb_kinetic_energy(cell, point), 2.0)
                  / SmoothMath::max(
                      _turb_dissipation_rate(cell, point), max_tol, 0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_IMPL_HPP
