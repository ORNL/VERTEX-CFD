#ifndef VERTEXCFD_CLOSURE_GODUNOVPOWELLSOURCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_GODUNOVPOWELLSOURCE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
GodunovPowellSource<EvalType, Traits, NumSpaceDim>::GodunovPowellSource(
    const panzer::IntegrationRule& ir,
    const MHDProperties::FullInductionMHDProperties& mhd_props)
    : _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
    , _divergence_total_magnetic_field("divergence_total_magnetic_field",
                                       ir.dl_scalar)
{
    // Evaluated fields
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _godunov_powell_momentum_source,
                                   "GODUNOV_POWELL_SOURCE_momentum_");
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _godunov_powell_induction_source,
                                   "GODUNOV_POWELL_SOURCE_induction_");

    // Dependent fields
    this->addDependentField(_divergence_total_magnetic_field);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _total_magnetic_field, "total_magnetic_field_");

    this->setName("Godunov-Powell Source " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void GodunovPowellSource<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
GodunovPowellSource<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _divergence_total_magnetic_field.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _godunov_powell_momentum_source[dim](cell, point)
                    = -_divergence_total_magnetic_field(cell, point)
                      * _total_magnetic_field[dim](cell, point)
                      / _magnetic_permeability;
                _godunov_powell_induction_source[dim](cell, point)
                    = -_divergence_total_magnetic_field(cell, point)
                      * _velocity[dim](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_GODUNOVPOWELLSOURCE_IMPL_HPP
