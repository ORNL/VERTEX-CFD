#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleKTauEddyViscosity<EvalType, Traits, NumSpaceDim>::
    IncompressibleKTauEddyViscosity(const panzer::IntegrationRule& ir)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);

    // Add evaluated fields
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("Incompressible K-Tau Turbulent Eddy Viscosity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleKTauEddyViscosity<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleKTauEddyViscosity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-20;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _nu_t(cell, point)
                = SmoothMath::max(
                      _turb_kinetic_energy(cell, point), max_tol, 0.0)
                  * SmoothMath::max(
                      _turb_specific_dissipation_rate(cell, point),
                      max_tol,
                      0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUEDDYVISCOSITY_IMPL_HPP
