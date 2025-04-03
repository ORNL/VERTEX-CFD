#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELOCALTIMESTEPSIZE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELOCALTIMESTEPSIZE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <utils/VertexCFD_Utils_SmoothMath.hpp>

#include <Panzer_HierarchicParallelism.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLocalTimeStepSize<EvalType, Traits, NumSpaceDim>::
    IncompressibleLocalTimeStepSize(const panzer::IntegrationRule& ir)
    : _local_dt("local_dt", ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
{
    // Add evaluated field
    this->addEvaluatedField(_local_dt);

    // Add dependent fields
    this->addDependentField(_element_length);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    this->setName("Incompressible Local Time Step Size");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLocalTimeStepSize<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLocalTimeStepSize<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _local_dt.extent(1);
    const int num_grad_dim = _element_length.extent(2);

    const double tol = 1.0e-8;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            scalar_type one_over_dt = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                one_over_dt += SmoothMath::abs(_velocity[dim](cell, point), tol)
                               / _element_length(cell, point, dim);
            }
            _local_dt(cell, point) = 1.0 / one_over_dt;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELOCALTIMESTEPSIZE_IMPL_HPP
