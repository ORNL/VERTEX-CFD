#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYVISCOSITY_IMPL_HPP

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
IncompressibleLSVOFEntropyViscosity<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFEntropyViscosity(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& equation_name,
        const Teuchos::RCP<panzer::GlobalData>& global_data)
    : _artificial_viscosity("artificial_viscosity", ir.dl_scalar)
    , _nu_max("ceiling_viscosity", ir.dl_scalar)
    , _nu_e("local_entropy", ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
    , _entropy_residual("entropy_residual", ir.dl_scalar)
    , _cmax(closure_params.get<double>("Ceiling viscosity parameter"))
    , _ce(closure_params.get<double>("Local entropy parameter"))
    , _global_data(global_data)
{
    // Evaluated fields
    this->addEvaluatedField(_artificial_viscosity);
    this->addEvaluatedField(_nu_max);
    this->addEvaluatedField(_nu_e);

    // Dependent fields
    this->addDependentField(_element_length);
    this->addDependentField(_entropy_residual);

    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    this->setName(equation_name + " Incompressible LSVOF Entropy Viscosity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFEntropyViscosity<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    const auto& pl = *_global_data->pl;
    const std::string entropy_norm_name = "Max Entropy - entropy_fluctuation";
    _entropy_norm = std::max(pl.getValue<EvalType>(entropy_norm_name), 1.0e-10);

    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_nu_max.get_view());
        bytes = scratch_view::shmem_size(_nu_max.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_nu_max.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFEntropyViscosity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using Kokkos::abs;
    using Kokkos::pow;
    using Kokkos::sqrt;
    using SmoothMath::max;
    using SmoothMath::min;

    const int cell = team.league_rank();
    const int num_point = _element_length.extent(1);
    const double max_tol = 1e-10;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_nu_max.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& mag_u = tmp_field(point, MAG_U);
            auto&& min_length = tmp_field(point, MIN_LENGTH);
            mag_u = 0.0;
            min_length = _element_length(cell, point, 0);

            for (int d = 0; d < num_space_dim; ++d)
            {
                mag_u += pow(_velocity[d](cell, point), 2.0);
                if (d > 0)
                    min_length = min(
                        _element_length(cell, point, d), min_length, 0.0);
            }

            _nu_max(cell, point) = _cmax * min_length
                                   * sqrt(max(mag_u, max_tol, 0.0));

            _nu_e(cell, point) = _ce * pow(min_length, 2.0)
                                 * abs(_entropy_residual(cell, point))
                                 / _entropy_norm;

            _artificial_viscosity(cell, point)
                = min(_nu_max(cell, point), _nu_e(cell, point), 0.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYVISCOSITY_IMPL_HPP
