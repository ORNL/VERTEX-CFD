#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLETIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLETIMEDERIVATIVE_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <string.h>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleVariableTimeDerivative<EvalType, Traits>::
    IncompressibleVariableTimeDerivative(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params)
    : _variable_name(closure_params.get<std::string>("Field Name"))
    , _equation_name(closure_params.get<std::string>("Equation Name"))
    , _dqdt_var_eq("DQDT_" + _equation_name, ir.dl_scalar)
    , _dxdt_var("DXDT_" + _variable_name, ir.dl_scalar)
{
    // Evaluated variable
    this->addEvaluatedField(_dqdt_var_eq);

    // Dependent variable
    this->addDependentField(_dxdt_var);

    // Closure model name
    const int num_grad_dim = ir.spatial_dimension;
    this->setName(_equation_name + " Incompressible Time Derivative "
                  + std::to_string(num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleVariableTimeDerivative<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleVariableTimeDerivative<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_var_eq.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _dqdt_var_eq(cell, point) = _dxdt_var(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEVARIABLETIMEDERIVATIVE_IMPL_HPP
