#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLENUSSELTNUMBER_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLENUSSELTNUMBER_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleNusseltNumber<EvalType, Traits>::IncompressibleNusseltNumber(
    const panzer::IntegrationRule& ir)
    : _nusselt_number("nusselt_number", ir.dl_scalar)
    , _grad_temperature("GRAD_temperature", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _num_space_dim(ir.spatial_dimension)
{
    // Add evaluated field
    this->addEvaluatedField(_nusselt_number);

    // Add dependent field
    this->addDependentField(_grad_temperature);
    this->addDependentField(_normals);

    this->setName("Incompressible Nusselt Number "
                  + std::to_string(_num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleNusseltNumber<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleNusseltNumber<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_temperature.extent(1);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             // Nusselt number calculated with VERTEX-CFD needs
                             // to be multiplied with
                             // k/k_{ref} * 1/(T_w-T_{inf})
                             // as the model that is used is
                             // Nu= k/k_{ref} * d(\theta)/d(n)
                             // where \theta = (T-T_{inf})/(T_w-T_{inf}).
                             _nusselt_number(cell, point) = 0;
                             for (int i = 0; i < _num_space_dim; ++i)
                             {
                                 _nusselt_number(cell, point)
                                     += _grad_temperature(cell, point, i)
                                        * _normals(cell, point, i);
                             }
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLENUSSELTNUMBER_IMPL_HPP
