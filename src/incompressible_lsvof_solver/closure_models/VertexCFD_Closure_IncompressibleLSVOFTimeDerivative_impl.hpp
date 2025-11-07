#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFTIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFTIMEDERIVATIVE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFTimeDerivative<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFTimeDerivative(const panzer::IntegrationRule& ir,
                                      const Teuchos::ParameterList& lsvof_params)
    : _dqdt_continuity("DQDT_continuity", ir.dl_scalar)
    , _rho("density", ir.dl_scalar)
    , _dxdt_pressure("DXDT_lagrange_pressure", ir.dl_scalar)
    , _dxdt_density("DXDT_density", ir.dl_scalar)
    , _beta(lsvof_params.get<double>("Mixture Artificial Compressibility"))
{
    // Evaluated fields
    this->addEvaluatedField(_dqdt_continuity);

    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _dqdt_momentum, "DQDT_momentum_");

    // Dependent fields
    this->addDependentField(_rho);
    this->addDependentField(_dxdt_pressure);
    this->addDependentField(_dxdt_density);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _dxdt_velocity, "DXDT_velocity_");

    this->setName("Incompressible LSVOF Time Derivative "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFTimeDerivative<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFTimeDerivative<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_continuity.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _dqdt_continuity(cell, point) = _dxdt_pressure(cell, point) / _beta;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _dqdt_momentum[dim](cell, point)
                    = _rho(cell, point) * _dxdt_velocity[dim](cell, point)
                      + _dxdt_density(cell, point)
                            * _velocity[dim](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFTIMEDERIVATIVE_IMPL_HPP
