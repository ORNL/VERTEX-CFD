#ifndef VERTEXCFD_CLOSURE_MHDVORTEXPROBLEMEXACT_IMPL_HPP
#define VERTEXCFD_CLOSURE_MHDVORTEXPROBLEMEXACT_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
MHDVortexProblemExact<EvalType, Traits, NumSpaceDim>::MHDVortexProblemExact(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& full_induction_params)
    : _ir_degree(ir.cubature_degree)
    , _lagrange_pressure("Exact_lagrange_pressure", ir.dl_scalar)
{
    const auto vel_0
        = full_induction_params.get<Teuchos::Array<double>>("velocity_0");
    const auto xy_0
        = full_induction_params.get<Teuchos::Array<double>>("center_0");
    for (int dim = 0; dim < 2; ++dim)
    {
        _vel_0[dim] = vel_0[dim];
        _xy_0[dim] = xy_0[dim];
    }

    // Add evaluated fields
    this->addEvaluatedField(_lagrange_pressure);
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _velocity, "Exact_velocity_");
    Utils::addEvaluatedVectorField(*this,
                                   ir.dl_scalar,
                                   _induced_magnetic_field,
                                   "Exact_induced_magnetic_field_");

    // Closure model name
    this->setName("MHD Vortex Problem Exact Solution "
                  + std::to_string(num_space_dim) + "D.");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MHDVortexProblemExact<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void MHDVortexProblemExact<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _time = workset.time;
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
MHDVortexProblemExact<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    using std::exp;
    const double exp_one = exp(1.0);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Coordinnates
            const double x = _ip_coords(cell, point, 0);
            const double y = _ip_coords(cell, point, 1);
            const double r2 = (x - _time * _vel_0[0] - _xy_0[0])
                                  * (x - _time * _vel_0[0] - _xy_0[0])
                              + (y - _time * _vel_0[0] - _xy_0[1])
                                    * (y - _time * _vel_0[0] - _xy_0[1]);

            // Exact solutions
            _lagrange_pressure(cell, point)
                = 1.0 + 0.5 * exp_one * (1.0 - r2 * exp(-r2));

            _induced_magnetic_field[0](cell, point) = exp(0.5 * (1.0 - r2))
                                                      * (_xy_0[1] - y);
            _induced_magnetic_field[1](cell, point) = exp(0.5 * (1.0 - r2))
                                                      * (x - _xy_0[0]);

            _velocity[0](cell, point) = _induced_magnetic_field[0](cell, point)
                                        + _vel_0[0];
            _velocity[1](cell, point) = _induced_magnetic_field[1](cell, point);

            if (num_space_dim == 3)
            {
                _induced_magnetic_field[2](cell, point) = 0.0;
                _velocity[2](cell, point) = 0.0;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_MHDVORTEXPROBLEMEXACT_IMPL_HPP
