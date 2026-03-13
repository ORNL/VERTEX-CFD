#ifndef VERTEXCFD_CLOSURE_PLASMASPECIESEOS_IMPL_HPP
#define VERTEXCFD_CLOSURE_PLASMASPECIESEOS_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"
#include "utils/VertexCFD_Utils_VelocityLayout.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
PlasmaSpeciesEOS<EvalType, Traits>::PlasmaSpeciesEOS(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& closure_params)
    : _species_name(closure_params.get<std::string>("Species Name"))
    , _ms(closure_params.get<double>("Species Mass"))
    , _p(_species_name + "_pressure", ir.dl_scalar)
    , _rho(_species_name + "_momentum_density", ir.dl_scalar)
    , _E(_species_name + "_energy", ir.dl_scalar)
    , _drhodt("DVDT_" + _species_name + "_momentum_density", ir.dl_scalar)
    , _dEdt("DVDT_" + _species_name + "_energy", ir.dl_scalar)
    , _nd(_species_name + "_number_density", ir.dl_scalar)
    , _velocity(_species_name + "_velocity",
                Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
    , _T(_species_name + "_temperature", ir.dl_scalar)
    , _dnddt("DXDT_" + _species_name + "_number_density", ir.dl_scalar)
    , _dveldt("DXDT_" + _species_name + "_velocity",
              Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
    , _dTdt("DXDT_" + _species_name + "_temperature", ir.dl_scalar)

{
    // Evaluated fields
    this->addEvaluatedField(_p);
    this->addEvaluatedField(_rho);
    this->addEvaluatedField(_E);
    this->addEvaluatedField(_drhodt);
    this->addEvaluatedField(_dEdt);

    // Dependent fields
    this->addDependentField(_nd);
    this->addDependentField(_velocity);
    this->addDependentField(_T);
    this->addDependentField(_dnddt);
    this->addDependentField(_dveldt);
    this->addDependentField(_dTdt);

    this->setName("Species EOS " + std::to_string(ir.spatial_dimension) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void PlasmaSpeciesEOS<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void PlasmaSpeciesEOS<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _nd.extent(1);
    const int num_vel_dim = _velocity.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Momentum density
            _rho(cell, point) = _ms * _nd(cell, point);
            _drhodt(cell, point) = _ms * _dnddt(cell, point);

            // Pressure
            _p(cell, point) = _nd(cell, point) * _T(cell, point);

            // Energy density
            _E(cell, point) = 0.0;
            _dEdt(cell, point) = 0.0;
            for (int dim = 0; dim < num_vel_dim; ++dim)
            {
                _E(cell, point) += _velocity(cell, point, dim)
                                   * _velocity(cell, point, dim);
                _dEdt(cell, point) += _rho(cell, point)
                                      * _velocity(cell, point, dim)
                                      * _dveldt(cell, point, dim);
                _dEdt(cell, point) += 0.5 * _drhodt(cell, point)
                                      * _velocity(cell, point, dim)
                                      * _velocity(cell, point, dim);
            }
            _E(cell, point) *= 0.5 * _rho(cell, point);
            _E(cell, point) += 0.5 * 3. * _p(cell, point);
            _dEdt(cell, point) += 0.5 * 3.
                                  * (_dnddt(cell, point) * _T(cell, point)
                                     + _nd(cell, point) * _dTdt(cell, point));
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_PLASMASPECIESEOS_IMPL_HPP
