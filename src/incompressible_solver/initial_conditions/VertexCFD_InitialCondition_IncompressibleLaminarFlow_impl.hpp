#ifndef VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLELAMINARFLOW_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLELAMINARFLOW_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Teuchos_Array.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::IncompressibleLaminarFlow(
    const Teuchos::ParameterList& ic_params,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const panzer::PureBasis& basis)
    : _lagrange_pressure("lagrange_pressure", basis.functional)
    , _temperature("temperature", basis.functional)
    , _basis_name(basis.name())
    , _solve_temp(fluid_prop.solveTemperature())
    , _min(ic_params.get<double>("Minimum height"))
    , _max(ic_params.get<double>("Maximum height"))
    , _vel_avg(ic_params.get<double>("Average velocity"))
    , _vel_max(num_space_dim == 2 ? 3.0 / 2.0 * _vel_avg : 2.0 * _vel_avg)
    , _T_init(std::numeric_limits<double>::quiet_NaN())
{
    this->addEvaluatedField(_lagrange_pressure);
    this->addUnsharedField(_lagrange_pressure.fieldTag().clone());

    Utils::addEvaluatedVectorField(
        *this, basis.functional, _velocity, "velocity_", true);

    if (_solve_temp)
    {
        _T_init = ic_params.get<double>("Temperature");
        this->addEvaluatedField(_temperature);
        this->addUnsharedField(_temperature.fieldTag().clone());
    }

    this->setName("Laminar Flow Initial Condition");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLaminarFlow<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _velocity[0].extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            double r2 = _basis_coords(cell, basis, 1)
                        * _basis_coords(cell, basis, 1);
            const double H = 0.5 * (_max - _min);
            _velocity[1](cell, basis) = 0.0;
            if (num_space_dim == 3)
            {
                r2 += _basis_coords(cell, basis, 2)
                      * _basis_coords(cell, basis, 2);
                _velocity[2](cell, basis) = 0.0;
            }
            _velocity[0](cell, basis) = _vel_max * (1.0 - r2 / (H * H));
            _lagrange_pressure(cell, basis) = 0.0;
            if (_solve_temp)
                _temperature(cell, basis) = _T_init;
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLELAMINARFLOW_IMPL_HPP
