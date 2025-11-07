#ifndef VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONLOCALTIMESTEPSIZE_IMPL_HPP
#define VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONLOCALTIMESTEPSIZE_IMPL_HPP

#include "utils/VertexCFD_Utils_MagneticLayout.hpp"
#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <cmath>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
SolidFullInductionLocalTimeStepSize<EvalType, Traits>::
    SolidFullInductionLocalTimeStepSize(
        const panzer::IntegrationRule& ir,
        const MHDProperties::FullInductionMHDProperties& mhd_props)
    : _local_dt("local_dt", ir.dl_scalar)
    , _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
    , _c_h(mhd_props.hyperbolicDivergenceCleaningSpeed())
    , _element_length("element_length", ir.dl_vector)
    , _solid_density("solid_density", ir.dl_scalar)
    , _total_magnetic_field(
          "total_magnetic_field",
          Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
{
    // Add evaluated field
    this->addEvaluatedField(_local_dt);

    // Add dependent fields
    this->addDependentField(_element_length);
    this->addDependentField(_solid_density);
    this->addDependentField(_total_magnetic_field);

    this->setName("Solid Full Induction Local Time Step Size");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void SolidFullInductionLocalTimeStepSize<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
SolidFullInductionLocalTimeStepSize<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _local_dt.extent(1);
    const int num_grad_dim = _element_length.extent(2);

    const double tol = 1.0e-8;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _local_dt(cell, point) = 0.0;
            const auto& rho = _solid_density(cell, point);
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                using Kokkos::sqrt;
                using SmoothMath::abs;
                using SmoothMath::max;
                _local_dt(cell, point)
                    += max(abs(_total_magnetic_field(cell, point, dim), tol)
                               / sqrt(_magnetic_permeability * rho),
                           _c_h,
                           tol)
                       / _element_length(cell, point, dim);
            }
            _local_dt(cell, point) = 1.0 / _local_dt(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SOLIDFULLINDUCTIONLOCALTIMESTEPSIZE_IMPL_HPP
