#ifndef VERTEXCFD_CLOSURE_INDUCTIONCONVECTIVEMOMENTUMFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INDUCTIONCONVECTIVEMOMENTUMFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_MagneticLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
InductionConvectiveMomentumFlux<EvalType, Traits, NumSpaceDim>::
    InductionConvectiveMomentumFlux(
        const panzer::IntegrationRule& ir,
        const MHDProperties::FullInductionMHDProperties& mhd_props,
        const std::string& flux_prefix)
    : _total_magnetic_field(
          "total_magnetic_field",
          Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
    , _magnetic_pressure("magnetic_pressure", ir.dl_scalar)
    , _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
{
    // Contributed fields
    Utils::addContributedVectorField(*this, ir.dl_vector, _momentum_flux,
                                   flux_prefix + "CONVECTIVE_FLUX_"
                                                 "momentum_");

    // Dependent fields
    this->addDependentField(_total_magnetic_field);
    this->addDependentField(_magnetic_pressure);

    this->setName("Induction Convective Momentum Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void InductionConvectiveMomentumFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
InductionConvectiveMomentumFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _momentum_flux[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int flux_dim = 0; flux_dim < num_space_dim; ++flux_dim)
            {
                for (int vec_dim = 0; vec_dim < num_space_dim; ++vec_dim)
                {
                    _momentum_flux[vec_dim](cell, point, flux_dim)
                        -= _total_magnetic_field(cell, point, flux_dim)
                           * _total_magnetic_field(cell, point, vec_dim)
                           / _magnetic_permeability;
                }
                // Add the magnetic pressure contribution to momentum flux.
                _momentum_flux[flux_dim](cell, point, flux_dim)
                    += _magnetic_pressure(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INDUCTIONCONVECTIVEMOMENTUMFLUX_IMPL_HPP
