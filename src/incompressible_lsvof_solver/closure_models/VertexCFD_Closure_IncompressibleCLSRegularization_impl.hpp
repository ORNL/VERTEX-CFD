#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSREGULARIZATION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSREGULARIZATION_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleCLSRegularization<EvalType, Traits, NumSpaceDim>::
    IncompressibleCLSRegularization(const panzer::IntegrationRule& ir)
    : _cls_reg("REG_level_set_equation", ir.dl_vector)
    , _lambda("CLS_lambda", ir.dl_scalar)
    , _grad_phi("GRAD_level_set", ir.dl_vector)
    , _q("CLS_interface_normal", ir.dl_vector)
{
    // Evaluated fields
    this->addEvaluatedField(_cls_reg);

    // Dependent fields
    this->addDependentField(_lambda);
    this->addDependentField(_grad_phi);
    this->addDependentField(_q);

    this->setName("Incompressible CLS Regularization "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleCLSRegularization<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleCLSRegularization<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _cls_reg.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _cls_reg(cell, point, dim)
                    = -1.0 * _lambda(cell, point)
                      * (_grad_phi(cell, point, dim) - _q(cell, point, dim));
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSREGULARIZATION_IMPL_HPP
