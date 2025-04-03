#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleSpalartAllmarasEddyViscosity<EvalType, Traits>::
    IncompressibleSpalartAllmarasEddyViscosity(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _sa_var("spalart_allmaras_variable", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _cv1(7.1)
    , _num_grad_dim(ir.spatial_dimension)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_sa_var);

    // Add evaluated fields
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("Spalart-Allmaras Incompressible Turbulent Eddy Viscosity "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleSpalartAllmarasEddyViscosity<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleSpalartAllmarasEddyViscosity<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const auto sa_var = _sa_var(cell, point);
            if (sa_var >= 0.0)
            {
                const scalar_type xi3 = pow(sa_var / _nu, 3.0);
                const scalar_type f_v1 = xi3 / (xi3 + _cv1 * _cv1 * _cv1);
                _nu_t(cell, point) = sa_var * f_v1;
            }
            else
            {
                _nu_t(cell, point) = 0.0;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASEDDYVISCOSITY_IMPL_HPP
