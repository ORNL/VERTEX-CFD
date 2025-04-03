#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLETIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLETIMEDERIVATIVE_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleTimeDerivative<EvalType, Traits, NumSpaceDim>::
    IncompressibleTimeDerivative(
        const panzer::IntegrationRule& ir,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _dqdt_continuity("DQDT_continuity", ir.dl_scalar)
    , _dqdt_energy("DQDT_energy", ir.dl_scalar)
    , _dxdt_lagrange_pressure("DXDT_lagrange_pressure", ir.dl_scalar)
    , _dxdt_temperature("DXDT_temperature", ir.dl_scalar)
    , _rho(fluid_prop.constantDensity())
    , _solve_temp(fluid_prop.solveTemperature())
    , _rhoCp(fluid_prop.constantHeatCapacity())
    , _beta(fluid_prop.artificialCompressibility())
{
    // Evaluated continuity
    this->addEvaluatedField(_dqdt_continuity);

    // Dependent and evaluated velocity-based fields
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _dqdt_momentum, "DQDT_momentum_");
    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _dxdt_velocity, "DXDT_velocity_");
    this->addDependentField(_dxdt_lagrange_pressure);

    // Dependent and evaluated temperature
    if (_solve_temp)
    {
        this->addEvaluatedField(_dqdt_energy);
        this->addDependentField(_dxdt_temperature);
    }

    this->setName("Incompressible Time Derivative "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTimeDerivative<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleTimeDerivative<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_continuity.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            _dqdt_continuity(cell, point)
                = _dxdt_lagrange_pressure(cell, point) / _beta;

            for (int dim = 0; dim < num_space_dim; ++dim)
                _dqdt_momentum[dim](cell, point)
                    = _rho * _dxdt_velocity[dim](cell, point);

            if (_solve_temp)
            {
                _dqdt_energy(cell, point) = _rhoCp
                                            * _dxdt_temperature(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLETIMEDERIVATIVE_IMPL_HPP
