#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFARTIFICIALCOMPRESSION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFARTIFICIALCOMPRESSION_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFArtificialCompression<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFArtificialCompression(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& scalar_name,
        const std::string& equation_name,
        const std::string& flux_prefix,
        const std::string& gradient_prefix,
        const std::string& field_prefix)
    : _scalar_flux(flux_prefix + "ARTIFICIAL_COMPRESSION_" + equation_name,
                   ir.dl_vector)
    , _interface_velocity("interface_velocity", ir.dl_vector)
    , _scalar(field_prefix + scalar_name, ir.dl_scalar)
    , _grad_scalar(gradient_prefix + "GRAD_" + scalar_name, ir.dl_vector)
    , _C_alpha(closure_params.get<double>("Compression Factor"))
{
    // Evaluated fields
    this->addEvaluatedField(_scalar_flux);
    this->addEvaluatedField(_interface_velocity);

    // Dependent fields
    this->addDependentField(_scalar);
    this->addDependentField(_grad_scalar);

    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _velocity, field_prefix + "velocity_");

    this->setName(equation_name
                  + " Incompressible LSVOF Artificial Compression "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFArtificialCompression<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size
            = Kokkos::dimension_scalar(_interface_velocity.get_view());
        bytes = scratch_view::shmem_size(
            _interface_velocity.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_interface_velocity.extent(1),
                                         NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFArtificialCompression<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using Kokkos::sqrt;
    using SmoothMath::max;
    using SmoothMath::min;
    const int cell = team.league_rank();
    const int num_point = _scalar_flux.extent(1);
    const double max_tol = 1e-10;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size
            = Kokkos::dimension_scalar(_interface_velocity.get_view());
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
            auto&& mag_grad_scalar = tmp_field(point, MAG_GRAD_SCALAR);
            mag_u = 0.0;
            mag_grad_scalar = 0.0;

            for (int d = 0; d < num_space_dim; ++d)
            {
                mag_u += pow(_velocity[d](cell, point), 2.0);
                mag_grad_scalar += pow(_grad_scalar(cell, point, d), 2.0);
            }

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _scalar_flux(cell, point, dim)
                    = _scalar(cell, point) * (1.0 - _scalar(cell, point));

                _scalar_flux(cell, point, dim) = min(
                    max(_scalar_flux(cell, point, dim), 0.0, 0.0), 1.0, 0.0);

                _interface_velocity(cell, point, dim)
                    = _C_alpha
                      * sqrt(mag_u / max(mag_grad_scalar, max_tol, 0.0))
                      * _grad_scalar(cell, point, dim);

                _scalar_flux(cell, point, dim)
                    *= _interface_velocity(cell, point, dim);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFARTIFICIALCOMPRESSION_IMPL_HPP
