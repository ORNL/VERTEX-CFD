#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSNONRECONSTRUCTEDNORMAL_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSNONRECONSTRUCTEDNORMAL_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleCLSNonReconstructedNormal<EvalType, Traits, NumSpaceDim>::
    IncompressibleCLSNonReconstructedNormal(const panzer::IntegrationRule& ir)
    : _q("CLS_interface_normal", ir.dl_vector)
    , _grad_phi_0("STAR_GRAD_level_set", ir.dl_vector)
{
    // Evaluated fields
    this->addEvaluatedField(_q);

    // Dependent fields
    this->addDependentField(_grad_phi_0);

    this->setName("Incompressible CLS Non-Reconstructed Interface Normal "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleCLSNonReconstructedNormal<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleCLSNonReconstructedNormal<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _q.extent(1);

    const double tol = 1e-15;
    const int N = num_space_dim - 1;

    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Store mag(grad(phi_0)) in _q[N]
            _q(cell, point, N) = 0.0;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _q(cell, point, N) += pow(_grad_phi_0(cell, point, dim), 2.0);
            }

            _q(cell, point, N) = pow(_q(cell, point, N) + tol, 0.5);

            // Evaluate interface normal
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _q(cell, point, dim) = _grad_phi_0(cell, point, dim)
                                       / _q(cell, point, N);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSNONRECONSTRUCTEDNORMAL_IMPL_HPP
