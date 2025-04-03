#ifndef VERTEXCFD_CLOSURE_MAGNETICPRESSURE_IMPL_HPP
#define VERTEXCFD_CLOSURE_MAGNETICPRESSURE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
MagneticPressure<EvalType, Traits>::MagneticPressure(
    const panzer::IntegrationRule& ir,
    const MHDProperties::FullInductionMHDProperties& mhd_props)
    : _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
    , _magnetic_pressure("magnetic_pressure", ir.dl_scalar)
{
    // Add dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _total_magnetic_field, "total_magnetic_field_");

    // Add evaluated fields
    this->addEvaluatedField(_magnetic_pressure);

    // Closure model name
    this->setName("Magnetic Pressure");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void MagneticPressure<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void MagneticPressure<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _magnetic_pressure.extent(1);
    const int num_field_dim = _total_magnetic_field.size();

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _magnetic_pressure(cell, point) = 0.0;
            for (int dim = 0; dim < num_field_dim; ++dim)
            {
                _magnetic_pressure(cell, point)
                    += _total_magnetic_field[dim](cell, point)
                       * _total_magnetic_field[dim](cell, point);
            }
            _magnetic_pressure(cell, point) /= 2.0 * _magnetic_permeability;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_MAGNETICPRESSURE_IMPL_HPP
