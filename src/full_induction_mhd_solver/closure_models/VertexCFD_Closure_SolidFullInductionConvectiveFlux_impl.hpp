#ifndef VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONCONVECTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONCONVECTIVEFLUX_IMPL_HPP

#include "utils/VertexCFD_Utils_MagneticLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
SolidFullInductionConvectiveFlux<EvalType, Traits, NumSpaceDim>::
    SolidFullInductionConvectiveFlux(
        const panzer::IntegrationRule& ir,
        const MHDProperties::FullInductionMHDProperties& mhd_props,
        const std::string& flux_prefix,
        const std::string& field_prefix)
    : _magnetic_correction_potential_flux(
          flux_prefix + "CONVECTIVE_FLUX_magnetic_correction_potential",
          ir.dl_vector)
    , _total_magnetic_field(
          "total_magnetic_field",
          Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
    , _scalar_magnetic_potential(field_prefix + "scalar_magnetic_potential",
                                 ir.dl_scalar)
    , _c_h(mhd_props.hyperbolicDivergenceCleaningSpeed())
{
    // Evaluated fields
    this->addEvaluatedField(_magnetic_correction_potential_flux);

    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _induction_flux,
                                 flux_prefix + "CONVECTIVE_FLUX_"
                                               "induction_");

    // Dependent fields
    this->addDependentField(_scalar_magnetic_potential);
    this->addDependentField(_total_magnetic_field);

    this->setName("Solid Full Induction Convective Flux "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void SolidFullInductionConvectiveFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
SolidFullInductionConvectiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _induction_flux[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int flux_dim = 0; flux_dim < num_space_dim; ++flux_dim)
            {
                _induction_flux[flux_dim](cell, point, flux_dim)
                    = _c_h * _scalar_magnetic_potential(cell, point);
                _magnetic_correction_potential_flux(cell, point, flux_dim)
                    = _c_h * _total_magnetic_field(cell, point, flux_dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONCONVECTIVEFLUX_IMPL_HPP
