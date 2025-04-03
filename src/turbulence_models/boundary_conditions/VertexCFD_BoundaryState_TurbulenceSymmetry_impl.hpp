#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCESYMMETRY_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCESYMMETRY_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// Symmetry boundary condition for turbulence quantities
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
TurbulenceSymmetry<EvalType, Traits>::TurbulenceSymmetry(
    const panzer::IntegrationRule& ir, const std::string variable_name)
    : _boundary_variable("BOUNDARY_" + variable_name, ir.dl_scalar)
    , _boundary_grad_variable("BOUNDARY_GRAD_" + variable_name, ir.dl_vector)
    , _variable(variable_name, ir.dl_scalar)
    , _grad_variable("GRAD_" + variable_name, ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _num_grad_dim(ir.spatial_dimension)
{
    // Add evaluated fields
    this->addEvaluatedField(_boundary_variable);
    this->addEvaluatedField(_boundary_grad_variable);

    // Add dependent fields
    this->addDependentField(_variable);
    this->addDependentField(_grad_variable);
    this->addDependentField(_normals);

    this->setName(variable_name + " Boundary State Turbulence Model Symmetry "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void TurbulenceSymmetry<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void TurbulenceSymmetry<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_variable.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Assign boundary values
            _boundary_variable(cell, point) = _variable(cell, point);

            // Compute boundary normal gradient
            scalar_type grad_var_dot_n = 0.0;

            for (int d = 0; d < _num_grad_dim; ++d)
            {
                grad_var_dot_n += _grad_variable(cell, point, d)
                                  * _normals(cell, point, d);
            }

            // Assign gradient
            for (int d = 0; d < _num_grad_dim; ++d)
            {
                _boundary_grad_variable(cell, point, d)
                    = _grad_variable(cell, point, d)
                      - grad_var_dot_n * _normals(cell, point, d);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_TURBULENCESYMMETRY_IMPL_HPP
