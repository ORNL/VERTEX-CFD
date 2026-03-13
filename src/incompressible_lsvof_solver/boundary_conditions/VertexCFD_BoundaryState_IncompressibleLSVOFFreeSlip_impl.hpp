#ifndef VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFFREESLIP_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFFREESLIP_IMPL_HPP

#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFFreeSlip<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFFreeSlip(const panzer::IntegrationRule& ir,
                                const int& num_lsvof_dofs,
                                const std::string& lsvof_model_name,
                                const std::string& continuity_model_name,
                                const bool& build_mom_equ)
    : _phase_layout(Utils::buildPhaseLayout(ir.dl_scalar, num_lsvof_dofs))
    , _boundary_lagrange_pressure("BOUNDARY_lagrange_pressure", ir.dl_scalar)
    , _boundary_grad_lagrange_pressure("BOUNDARY_GRAD_lagrange_pressure",
                                       ir.dl_vector)
    , _boundary_alphas("BOUNDARY_volume_fractions", _phase_layout)
    , _boundary_phi("BOUNDARY_level_set", ir.dl_scalar)
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _alphas("volume_fractions", _phase_layout)
    , _phi("level_set", ir.dl_scalar)
    , _lsvof_model_type(LSVOFModelType::VOF)
    , _is_edac(continuity_model_name == "EDAC" ? true : false)
    , _build_mom_equ(build_mom_equ)
{
    // Check LSVOF model type
    if (lsvof_model_name == "CLS")
        _lsvof_model_type = LSVOFModelType::CLS;

    // Evaluated fields
    if (_build_mom_equ)
    {
        this->addEvaluatedField(_boundary_lagrange_pressure);
        if (_is_edac)
            this->addEvaluatedField(_boundary_grad_lagrange_pressure);

        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_vector,
                                       _boundary_grad_velocity,
                                       "BOUNDARY_GRAD_velocity_");
    }

    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _boundary_velocity, "BOUNDARY_velocity_");

    if (_lsvof_model_type == LSVOFModelType::VOF)
    {
        this->addEvaluatedField(_boundary_alphas);
    }
    else if (_lsvof_model_type == LSVOFModelType::CLS)
    {
        this->addEvaluatedField(_boundary_phi);
    }

    // Dependent fields
    this->addDependentField(_normals);

    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    if (_build_mom_equ)
    {
        this->addDependentField(_lagrange_pressure);
        if (_is_edac)
            this->addDependentField(_grad_lagrange_pressure);

        Utils::addDependentVectorField(
            *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");
    }

    if (_lsvof_model_type == LSVOFModelType::VOF)
    {
        this->addDependentField(_alphas);
    }
    else if (_lsvof_model_type == LSVOFModelType::CLS)
    {
        this->addDependentField(_phi);
    }

    this->setName("Boundary State Incompressible LSVOF No-Slip");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFFreeSlip<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFFreeSlip<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _boundary_lagrange_pressure.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Set no-penetration condition for velocity
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                _boundary_velocity[vel_dim](cell, point)
                    = _velocity[vel_dim](cell, point);

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    _boundary_velocity[vel_dim](cell, point)
                        -= _velocity[dim](cell, point)
                           * _normals(cell, point, dim)
                           * _normals(cell, point, vel_dim);
                }
            }

            // Apply symmetry conditions to NS eqn quantities
            if (_build_mom_equ)
            {
                _boundary_lagrange_pressure(cell, point)
                    = _lagrange_pressure(cell, point);

                for (int d = 0; d < num_space_dim; ++d)
                {
                    // Initialize gradients to interior values
                    if (_is_edac)
                    {
                        _boundary_grad_lagrange_pressure(cell, point, d)
                            = _grad_lagrange_pressure(cell, point, d);
                    }

                    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
                    {
                        _boundary_grad_velocity[vel_dim](cell, point, d)
                            = _grad_velocity[vel_dim](cell, point, d);

                        // Subtract wall-normal components
                        for (int grad_dim = 0; grad_dim < num_space_dim;
                             ++grad_dim)
                        {
                            _boundary_grad_velocity[vel_dim](cell, point, d)
                                -= _grad_velocity[vel_dim](
                                       cell, point, grad_dim)
                                   * _normals(cell, point, grad_dim)
                                   * _normals(cell, point, d);
                        }

                        if (_is_edac)
                        {
                            _boundary_grad_lagrange_pressure(cell, point, d)
                                -= _grad_lagrange_pressure(cell, point, vel_dim)
                                   * _normals(cell, point, vel_dim)
                                   * _normals(cell, point, d);
                        }
                    }
                }
            }

            // Set boundary LSVOF fields
            if (_lsvof_model_type == LSVOFModelType::VOF)
            {
                for (size_t phase = 0; phase < _alphas.extent(2); ++phase)
                {
                    _boundary_alphas(cell, point, phase)
                        = _alphas(cell, point, phase);
                }
            }
            else if (_lsvof_model_type == LSVOFModelType::CLS)
            {
                _boundary_phi(cell, point) = _phi(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_INCOMPRESSIBLELSVOFFREESLIP_IMPL_HPP
