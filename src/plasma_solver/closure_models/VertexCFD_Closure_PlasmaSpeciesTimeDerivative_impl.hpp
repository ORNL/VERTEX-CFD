#ifndef VERTEXCFD_CLOSURE_SPECIESTIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_SPECIESTIMEDERIVATIVE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"
#include "utils/VertexCFD_Utils_VelocityLayout.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
PlasmaSpeciesTimeDerivative<EvalType, Traits, NumSpaceDim>::
    PlasmaSpeciesTimeDerivative(const panzer::IntegrationRule& ir,
                                const std::string species_name)
    : _dqdt_continuity("DQDT_" + species_name + "_number_density_equation",
                       ir.dl_scalar)
    , _dqdt_energy("DQDT_" + species_name + "_energy_equation", ir.dl_scalar)
    , _velocity(species_name + "_velocity",
                Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
    , _rho(species_name + "_momentum_density", ir.dl_scalar)
    , _dxdt_nd("DXDT_" + species_name + "_number_density", ir.dl_scalar)
    , _dxdt_velocity(
          "DXDT_" + species_name + "_velocity",
          Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
    , _dvdt_rho("DVDT_" + species_name + "_momentum_density", ir.dl_scalar)
    , _dvdt_E("DVDT_" + species_name + "_energy", ir.dl_scalar)
{
    // Evaluated fields
    this->addEvaluatedField(_dqdt_continuity);
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _dqdt_momentum,
                                   "DQDT_" + species_name + "_momentum_");
    this->addEvaluatedField(_dqdt_energy);

    // Dependent fields
    this->addDependentField(_rho);
    this->addDependentField(_velocity);
    this->addDependentField(_dxdt_nd);
    this->addDependentField(_dxdt_velocity);
    this->addDependentField(_dvdt_rho);
    this->addDependentField(_dvdt_E);

    this->setName(species_name + " Time Derivative "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void PlasmaSpeciesTimeDerivative<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
PlasmaSpeciesTimeDerivative<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_continuity.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Number density equation
            _dqdt_continuity(cell, point) = _dxdt_nd(cell, point);

            // Momentum equations
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _dqdt_momentum[dim](cell, point)
                    = _velocity(cell, point, dim) * _dvdt_rho(cell, point)
                      + _rho(cell, point) * _dxdt_velocity(cell, point, dim);
            }

            // Energy equation
            _dqdt_energy(cell, point) = _dvdt_E(cell, point);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SPECIESTIMEDERIVATIVE_IMPL_HPP
