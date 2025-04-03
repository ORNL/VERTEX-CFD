#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleSpalartAllmarasDiffusivityCoefficient<EvalType, Traits>::
    IncompressibleSpalartAllmarasDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _sa_var("spalart_allmaras_variable", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _cn1(16.0)
    , _sigma(2.0 / 3.0)
    , _one(1.0)
    , _num_grad_dim(ir.spatial_dimension)
    , _diffusivity_var("diffusivity_spalart_allmaras_variable", ir.dl_scalar)
{
    // Add dependent fields
    this->addDependentField(_sa_var);

    // Add evaluated fields
    this->addEvaluatedField(_diffusivity_var);

    // Closure model name
    this->setName("Spalart-Allmaras Incompressible Diffusivity Coefficient "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleSpalartAllmarasDiffusivityCoefficient<EvalType, Traits>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleSpalartAllmarasDiffusivityCoefficient<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _diffusivity_var.extent(1);
    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const scalar_type xi3 = pow(_sa_var(cell, point) / _nu, 3.0);
            const scalar_type f_n = _sa_var(cell, point) < 0.0
                                        ? (_cn1 + xi3) / (_cn1 - xi3)
                                        : _one;

            _diffusivity_var(cell, point) = (_nu + _sa_var(cell, point) * f_n)
                                            / _sigma;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_IMPL_HPP
