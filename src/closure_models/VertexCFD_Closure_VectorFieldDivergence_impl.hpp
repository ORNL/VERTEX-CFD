#ifndef VERTEXCFD_CLOSURE_VECTORFIELDDIVERGENCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_VECTORFIELDDIVERGENCE_IMPL_HPP

#include <utils/VertexCFD_Utils_SmoothMath.hpp>
#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
VectorFieldDivergence<EvalType, Traits, NumSpaceDim>::VectorFieldDivergence(
    const panzer::IntegrationRule& ir,
    const std::string& field_name,
    const std::string& closure_name)
    : _num_grad_dim(ir.spatial_dimension)
    , _use_abs(closure_name == "AbsVectorFieldDivergence" ? true : false)
    , _vector_field_divergence(
          (_use_abs ? "abs_divergence_" : "divergence_") + field_name,
          ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_vector_field_divergence);

    // Dependent fields
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_vector_field, "GRAD_" + field_name + "_");

    this->setName("Vector Field Divergence " + std::to_string(_num_grad_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void VectorFieldDivergence<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
VectorFieldDivergence<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_vector_field[0].extent(1);
    const double abs_tol = 1.0e-12;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _vector_field_divergence(cell, point) = 0.0;
            for (int d = 0; d < _num_grad_dim; ++d)
            {
                _vector_field_divergence(cell, point)
                    += _grad_vector_field[d](cell, point, d);
            }
            if (_use_abs)
            {
                _vector_field_divergence(cell, point) = SmoothMath::abs(
                    _vector_field_divergence(cell, point), abs_tol);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_VECTORFIELDDIVERGENCE_IMPL_HPP
