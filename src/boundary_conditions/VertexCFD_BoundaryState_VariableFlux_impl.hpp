#ifndef VERTEXCFD_BOUNDARYSTATE_VARIABLEFLUX_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_VARIABLEFLUX_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <algorithm>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
VariableFlux<EvalType, Traits>::VariableFlux(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& bc_params,
    const std::string variable_name,
    const std::string coefficient_name)
    : _boundary_variable("BOUNDARY_" + variable_name, ir.dl_scalar)
    , _boundary_grad_variable("BOUNDARY_GRAD_" + variable_name, ir.dl_vector)
    , _variable(variable_name, ir.dl_scalar)
    , _grad_variable("GRAD_" + variable_name, ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _dependent(coefficient_name, ir.dl_scalar)
    , _flux_dependence(!coefficient_name.empty())
{
    _flux = bc_params.get<double>("Flux Value");

    // Evaluated fields
    this->addEvaluatedField(_boundary_variable);
    this->addEvaluatedField(_boundary_grad_variable);

    // Dependent fields
    this->addDependentField(_variable);
    this->addDependentField(_grad_variable);
    this->addDependentField(_normals);
    if (_flux_dependence)
        this->addDependentField(_dependent);

    // Closure model names
    this->setName("Boundary State Constant Flux");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void VariableFlux<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void VariableFlux<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_variable.extent(1);
    const int num_grad_dim = _boundary_grad_variable.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _boundary_variable(cell, point) = _variable(cell, point);
            for (int d = 0; d < num_grad_dim; ++d)
            {
                _boundary_grad_variable(cell, point, d)
                    = _grad_variable(cell, point, d);

                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    _boundary_grad_variable(cell, point, d)
                        -= (_grad_variable(cell, point, dim)
                            * _normals(cell, point, dim))
                           * _normals(cell, point, d);
                }
                if (_flux_dependence)
                {
                    _boundary_grad_variable(cell, point, d)
                        += (_flux / _dependent(cell, point))
                           * _normals(cell, point, d);
                }
                else
                {
                    _boundary_grad_variable(cell, point, d)
                        += _flux * _normals(cell, point, d);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VARIABLEFLUX_IMPL_HPP
