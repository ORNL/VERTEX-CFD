#ifndef VERTEXCFD_CLOSURE_TOTALMAGNETICFIELD_IMPL_HPP
#define VERTEXCFD_CLOSURE_TOTALMAGNETICFIELD_IMPL_HPP

#include "utils/VertexCFD_Utils_MagneticLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
TotalMagneticField<EvalType, Traits, NumSpaceDim>::TotalMagneticField(
    const panzer::IntegrationRule& ir, const std::string& field_prefix)
    : _total_magnetic_field(
          "total_magnetic_field",
          Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
{
    // Add dependent fields
    Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _induced_magnetic_field,
                                   field_prefix + "induced_magnetic_field_");
    Utils::addDependentVectorField(*this,
                                   ir.dl_scalar,
                                   _external_magnetic_field,
                                   "external_magnetic_field_");
    // Add evaluated fields
    this->addEvaluatedField(_total_magnetic_field);

    // Closure model name
    this->setName("Total Magnetic Field " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void TotalMagneticField<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
TotalMagneticField<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _total_magnetic_field.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _total_magnetic_field(cell, point, dim)
                    = _induced_magnetic_field[dim](cell, point)
                      + _external_magnetic_field[dim](cell, point);
            }

            for (int dim = num_space_dim; dim < num_magnetic_field_dim; ++dim)
            {
                _total_magnetic_field(cell, point, dim)
                    = _external_magnetic_field[dim](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_TOTALMAGNETICFIELD_IMPL_HPP
