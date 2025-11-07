#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFERRORNORMS_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFERRORNORMS_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFErrorNorms<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFErrorNorms(const panzer::IntegrationRule& ir,
                                  const Teuchos::ParameterList& closure_params)
    : _L1_error_continuity("L1_Error_continuity", ir.dl_scalar)
    , _L2_error_continuity("L2_Error_continuity", ir.dl_scalar)
    , _volume("volume", ir.dl_scalar)
    , _dof_name(closure_params.get<std::string>("DOF Name"))
    , _exact_pressure("Exact_lagrange_pressure", ir.dl_scalar)
    , _exact_dof("Exact_" + _dof_name, ir.dl_scalar)
    , _pressure("lagrange_pressure", ir.dl_scalar)
    , _dof(_dof_name, ir.dl_scalar)
    , _L1_error_dof("L1_Error_" + _dof_name, ir.dl_scalar)
    , _L2_error_dof("L2_Error_" + _dof_name, ir.dl_scalar)
{
    //  Check if NS equations are solved
    _solve_ns = closure_params.isType<bool>("Compute NS Error Norms")
                    ? closure_params.get<bool>("Compute NS Error Norms")
                    : false;

    // Add NS terms
    if (_solve_ns)
    {
        // Dependent fields
        this->addDependentField(_exact_pressure);
        this->addDependentField(_pressure);

        Utils::addDependentVectorField(
            *this, ir.dl_scalar, _exact_velocity, "Exact_velocity_");
        Utils::addDependentVectorField(
            *this, ir.dl_scalar, _velocity, "velocity_");

        // Evaluated fields
        this->addEvaluatedField(_L1_error_continuity);
        this->addEvaluatedField(_L2_error_continuity);

        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, _L1_error_momentum, "L1_Error_momentum_");
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, _L2_error_momentum, "L2_Error_momentum_");
    }

    // LSVOF terms
    this->addDependentField(_dof);
    this->addDependentField(_exact_dof);
    this->addEvaluatedField(_L1_error_dof);
    this->addEvaluatedField(_L2_error_dof);

    this->addEvaluatedField(_volume);

    this->setName("Incompressible LSVOF Error Norms "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFErrorNorms<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFErrorNorms<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _volume.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using Kokkos::abs;
            using Kokkos::pow;

            // L1/L2 error norms for NS equations
            if (_solve_ns)
            {
                _L1_error_continuity(cell, point) = abs(
                    _pressure(cell, point) - _exact_pressure(cell, point));

                _L2_error_continuity(cell, point) = pow(
                    _pressure(cell, point) - _exact_pressure(cell, point), 2.0);

                for (int i = 0; i < num_space_dim; ++i)
                {
                    _L1_error_momentum[i](cell, point)
                        = abs(_velocity[i](cell, point)
                              - _exact_velocity[i](cell, point));
                    _L2_error_momentum[i](cell, point)
                        = pow(_velocity[i](cell, point)
                                  - _exact_velocity[i](cell, point),
                              2.0);
                }
            }

            // L1/L2 error norms for LSVOF terms
            _L1_error_dof(cell, point)
                = abs(_dof(cell, point) - _exact_dof(cell, point));
            _L2_error_dof(cell, point)
                = pow(_dof(cell, point) - _exact_dof(cell, point), 2.0);

            _volume(cell, point) = 1.0;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFERRORNORMS_IMPL_HPP
