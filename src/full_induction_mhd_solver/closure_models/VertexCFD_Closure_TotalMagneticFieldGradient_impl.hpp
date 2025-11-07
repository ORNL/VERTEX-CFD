#ifndef VERTEXCFD_CLOSURE_TOTALMAGNETICFIELDGRADIENT_IMPL_HPP
#define VERTEXCFD_CLOSURE_TOTALMAGNETICFIELDGRADIENT_IMPL_HPP

#include "utils/VertexCFD_Utils_MagneticLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
TotalMagneticFieldGradient<EvalType, Traits, NumSpaceDim>::TotalMagneticFieldGradient(
    const panzer::IntegrationRule& ir, const std::string& gradient_prefix)
    : _uniform_external_field(true)
    , _grad_total_magnetic_field(
          gradient_prefix + "GRAD_total_magnetic_field",
          Utils::buildMagneticGradLayout(ir.dl_vector, num_magnetic_field_dim))
    , _divergence_total_magnetic_field(
          gradient_prefix + "divergence_total_magnetic_field", ir.dl_scalar)
{
    // Add dependent fields
    Utils::addDependentVectorField(
        *this,
        ir.dl_vector,
        _grad_induced_magnetic_field,
        gradient_prefix + "GRAD_induced_magnetic_field_");
    if (!_uniform_external_field)
    {
        Utils::addDependentVectorField(*this,
                                       ir.dl_vector,
                                       _grad_external_magnetic_field,
                                       "GRAD_external_magnetic_field_");
    }

    // Add evaluated fields
    this->addEvaluatedField(_grad_total_magnetic_field);
    this->addEvaluatedField(_divergence_total_magnetic_field);

    // Closure model name
    this->setName("Total Magnetic Field Gradient"
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void TotalMagneticFieldGradient<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
TotalMagneticFieldGradient<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_total_magnetic_field.extent(1);
    const int num_grad_dim = _grad_total_magnetic_field.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int field_dim = 0; field_dim < num_space_dim; ++field_dim)
            {
                for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
                {
                    _grad_total_magnetic_field(cell, point, grad_dim, field_dim)
                        = _grad_induced_magnetic_field[field_dim](
                            cell, point, grad_dim);
                }
            }

            for (int field_dim = num_space_dim;
                 field_dim < num_magnetic_field_dim;
                 ++field_dim)
            {
                for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
                {
                    _grad_total_magnetic_field(cell, point, grad_dim, field_dim)
                        = 0.0;
                }
            }

            if (!_uniform_external_field)
            {
                for (int field_dim = 0; field_dim < num_magnetic_field_dim;
                     ++field_dim)
                {
                    for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
                    {
                        _grad_total_magnetic_field(
                            cell, point, grad_dim, field_dim)
                            += _grad_external_magnetic_field[field_dim](
                                cell, point, grad_dim);
                    }
                }
            }

            _divergence_total_magnetic_field(cell, point) = 0.0;
            for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
            {
                _divergence_total_magnetic_field(cell, point)
                    += _grad_total_magnetic_field(
                        cell, point, grad_dim, grad_dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_TOTALMAGNETICFIELDGRADIENT_IMPL_HPP
