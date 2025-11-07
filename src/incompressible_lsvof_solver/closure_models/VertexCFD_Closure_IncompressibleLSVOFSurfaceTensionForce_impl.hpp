#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSURFACETENSIONFORCE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSURFACETENSIONFORCE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorFieldView.hpp"

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
IncompressibleLSVOFSurfaceTensionForce<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFSurfaceTensionForce(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::vector<std::string>& phase_names,
        const std::string& flux_prefix,
        const std::string& gradient_prefix)
    : _sigma_alpha(closure_params.get<double>("Surface Tension Coefficient"))
    , num_phases(closure_params.get<int>("Number of Phases") - 1)
{
    // Evaluated fields
    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _surface_tension_flux,
				flux_prefix + "SURFACE_TENSION_FLUX_"
				"momentum_");

    // Dependent fields
    this->setName("Incompressible LSVOF Surface Tension Force "
                  + std::to_string(num_space_dim) + "D");

    Utils::addDependentVectorFieldView(*this,
                                       ir.dl_vector,
                                       num_phases,
                                       _phase_grad,
                                       gradient_prefix + "GRAD_",
                                       phase_names);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFSurfaceTensionForce<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size
            = Kokkos::dimension_scalar(_surface_tension_flux[0].get_view());
        bytes = scratch_view::shmem_size(_surface_tension_flux[0].extent(1),
                                         fad_size);

        bytes += scratch_view_Fs::shmem_size(
            _surface_tension_flux[0].extent(1), fad_size);

        bytes += scratch_view_normal::shmem_size(
            _surface_tension_flux[0].extent(1), num_space_dim, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_surface_tension_flux[0].extent(1));

        bytes
            += scratch_view_Fs::shmem_size(_surface_tension_flux[0].extent(1));

        bytes += scratch_view_normal::shmem_size(
            _surface_tension_flux[0].extent(1), num_space_dim);
    }
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFSurfaceTensionForce<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using Kokkos::sqrt;

    const int cell = team.league_rank();
    const int num_point = _surface_tension_flux[0].extent(1);
    const double max_tol = 1e-10;

    scratch_view scratch_data;
    scratch_view_Fs scratch_data_force;
    scratch_view_normal scratch_data_normal;

    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size
            = Kokkos::dimension_scalar(_surface_tension_flux[0].get_view());

        scratch_data = scratch_view(team.team_shmem(), num_point, fad_size);

        scratch_data_force
            = scratch_view_Fs(team.team_shmem(), num_point, fad_size);

        scratch_data_normal = scratch_view_normal(
            team.team_shmem(), num_point, num_space_dim, fad_size);
    }
    else
    {
        scratch_data = scratch_view(team.team_shmem(), num_point);

        scratch_data_force = scratch_view_Fs(team.team_shmem(), num_point);

        scratch_data_normal
            = scratch_view_normal(team.team_shmem(), num_point, num_space_dim);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& tmp_surf_force = scratch_data_force(point);
            auto&& mag_grad_scalar = scratch_data(point);
            auto&& unit_normal
                = Kokkos::subview(scratch_data_normal, point, Kokkos::ALL);

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                for (int d = 0; d < num_space_dim; ++d)
                {
                    _surface_tension_flux[d](cell, point, dim) = 0.0;
                }
            }

            for (int n = 0; n < num_phases; ++n)
            {
                mag_grad_scalar = 0.0;

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    unit_normal[dim] = 0.0;
                    mag_grad_scalar
                        += pow(_phase_grad[n](cell, point, dim), 2.0);
                }
                mag_grad_scalar
                    = sqrt(SmoothMath::max(mag_grad_scalar, max_tol, 0));

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    unit_normal[dim] = _phase_grad[n](cell, point, dim)
                                       / mag_grad_scalar;
                }

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    for (int d = 0; d < num_space_dim; ++d)
                    {
                        tmp_surf_force = -unit_normal[dim] * unit_normal[d];

                        if (d == dim)
                        {
                            tmp_surf_force += 1.0;
                        }

                        tmp_surf_force *= _sigma_alpha * mag_grad_scalar;

                        _surface_tension_flux[d](cell, point, dim)
                            += tmp_surf_force;
                    }
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSURFACETENSIONFORCE_IMPL_HPP
