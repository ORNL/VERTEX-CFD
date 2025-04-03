#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGAEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGAEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleKOmegaEddyViscosity<EvalType, Traits, NumSpaceDim>::
    IncompressibleKOmegaEddyViscosity(const panzer::IntegrationRule& ir)
    : _turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
    , _turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                      ir.dl_scalar)
    , _C_lim(7.0 / 8.0)
    , _beta_star(0.09)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_turb_kinetic_energy);
    this->addDependentField(_turb_specific_dissipation_rate);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("Incompressible K-Omega Turbulent Eddy Viscosity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleKOmegaEddyViscosity<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleKOmegaEddyViscosity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using std::pow;
    using std::sqrt;

    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const double max_tol = 1.0e-10;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            scalar_type Sij_Sij = 0.0;
            for (int i = 0; i < num_space_dim; ++i)
            {
                Sij_Sij += pow(_grad_velocity[i](cell, point, i), 2.0);
                for (int j = i + 1; j < num_space_dim; ++j)
                {
                    Sij_Sij += 0.5
                               * pow(_grad_velocity[i](cell, point, j)
                                         + _grad_velocity[j](cell, point, i),
                                     2.0);
                }
            }
            _nu_t(cell, point)
                = _turb_kinetic_energy(cell, point)
                  / SmoothMath::max(
                      SmoothMath::max(
                          _turb_specific_dissipation_rate(cell, point),
                          _C_lim * sqrt(2.0 * Sij_Sij / _beta_star),
                          0.0),
                      max_tol,
                      0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGAEDDYVISCOSITY_IMPL_HPP
