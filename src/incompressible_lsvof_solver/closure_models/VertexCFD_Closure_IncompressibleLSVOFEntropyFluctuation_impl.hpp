#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFLUCTUATION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFLUCTUATION_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFEntropyFluctuation<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFEntropyFluctuation(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& equation_name,
        const Teuchos::RCP<panzer::GlobalData>& global_data)
    : _entropy_fluctuation("entropy_fluctuation", ir.dl_scalar)
    , _entropy_function("entropy_function", ir.dl_scalar)
    , _domain_volume(closure_params.get<double>("Domain Volume"))
    , _global_data(global_data)
{
    // Evaluated fields
    this->addEvaluatedField(_entropy_fluctuation);

    // Dependent fields
    this->addDependentField(_entropy_function);

    this->setName(equation_name + " Incompressible LSVOF Entropy Fluctuation "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFEntropyFluctuation<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    const auto& pl = *_global_data->pl;
    const std::string entropy_int_name = "Entropy - entropy_function";
    _entropy_int = pl.getValue<EvalType>(entropy_int_name);

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFEntropyFluctuation<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using SmoothMath::abs;
    const int cell = team.league_rank();
    const int num_point = _entropy_function.extent(1);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_point),
                         [&](const int point) {
                             _entropy_fluctuation(cell, point)
                                 = abs((_entropy_function(cell, point)
                                        - (_entropy_int / _domain_volume)),
                                       0.0);
                         });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFLUCTUATION_IMPL_HPP
