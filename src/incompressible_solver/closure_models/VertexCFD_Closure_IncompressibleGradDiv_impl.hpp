#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEGRADDIV_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEGRADDIV_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleGradDiv<EvalType, Traits, NumSpaceDim>::IncompressibleGradDiv(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& stability_param_list,
    const std::string& flux_prefix,
    const std::string& gradient_prefix)
    : _gamma(stability_param_list.get<double>("GradDiv Stabilization "
                                              "Coefficient"))

{
    // Add evaludated/contributed fields
    Utils::addContributedVectorField(*this, ir.dl_vector, _momentum_flux,
                                 flux_prefix + "VISCOUS_FLUX_"
                                               "momentum_");

    // Add dependent fields
    Utils::addDependentVectorField(*this,
                                   ir.dl_vector,
                                   _grad_velocity,
                                   gradient_prefix + "GRAD_velocity_");

    this->setName("Incompressible Grad-Div Source "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleGradDiv<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleGradDiv<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _momentum_flux[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate weak form of grad(\gamma * div U)
            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    _momentum_flux[i](cell, point, i)
                        += _gamma * _grad_velocity[j](cell, point, j);
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLEGRADDIV_IMPL_HPP
