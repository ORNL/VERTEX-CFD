#ifndef VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALFIXED_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALFIXED_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
// This function should be used for Dirichlet boundary conditions. A ramping
// in time can be enabled.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ElectricPotentialFixed<EvalType, Traits>::ElectricPotentialFixed(
    const panzer::IntegrationRule& ir, const Teuchos::ParameterList& bc_params)
    : _boundary_electric_potential("BOUNDARY_electric_potential", ir.dl_scalar)
    , _boundary_grad_electric_potential("BOUNDARY_GRAD_electric_potential",
                                        ir.dl_vector)
    , _num_grad_dim(ir.spatial_dimension)
    , _grad_electric_potential("GRAD_electric_potential", ir.dl_vector)
{
    // Calculate the coefficients 'a' and 'b' for the linear time ramping
    // sc(t) = a * t + b
    _time_init = bc_params.isType<double>("Time Initial")
                     ? bc_params.get<double>(
                           "Time "
                           "Initial")
                     : 0.0;
    _time_final = bc_params.isType<double>("Time Final")
                      ? bc_params.get<double>("Time Final")
                      : 1.0E-06;
    const double dt = _time_final - _time_init;
    const auto sc_final = bc_params.get<double>("Final Value");
    const auto sc_init = bc_params.isType<double>("Initial Value")
                             ? bc_params.get<double>("Initial Value")
                             : sc_final;
    _a_sc = (sc_final - sc_init) / dt;
    _b_sc = sc_init - _a_sc * _time_init;

    // Add evaluated fields
    this->addEvaluatedField(_boundary_electric_potential);
    this->addEvaluatedField(_boundary_grad_electric_potential);

    // Add dependent fields
    this->addDependentField(_grad_electric_potential);

    this->setName("Boundary State Electric Potential Fixed "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ElectricPotentialFixed<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Get time and make sure it only varies between '_time_init' and
    // '_time_final'
    _time = std::max(workset.time, _time_init);
    _time = std::min(_time, _time_final);

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
ElectricPotentialFixed<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_electric_potential.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Assign time-dependent boundary values
            _boundary_electric_potential(cell, point) = _a_sc * _time + _b_sc;

            // Assign gradient.
            for (int d = 0; d < _num_grad_dim; ++d)
            {
                _boundary_grad_electric_potential(cell, point, d)
                    = _grad_electric_potential(cell, point, d);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_ELECTRICPOTENTIALFIXED_IMPL_HPP
