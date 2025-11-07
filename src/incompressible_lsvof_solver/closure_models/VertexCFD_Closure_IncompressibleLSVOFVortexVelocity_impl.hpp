#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVORTEXVELOCITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVORTEXVELOCITY_IMPL_HPP

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFVortexVelocity<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFVortexVelocity(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params)
    : _vel_0("velocity_0", ir.dl_scalar)
    , _vel_1("velocity_1", ir.dl_scalar)
    , _direction(1.0)
    , _ir_degree(ir.cubature_degree)
{
    // Check that simulation is 2D
    if (num_space_dim != 2)
    {
        const std::string msg
            = "ERROR: Incompressible LSVOF Vortex Velocity must be used for "
              "2D problem.\n";

        throw std::runtime_error(msg);
    }

    // Check vortex direction
    if (closure_params.isType<std::string>("Direction"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<Direction>(
                Teuchos::tuple<std::string>("Forward", "Reverse"), "Forward"));

        const Direction direction = type_validator->getIntegralValue(
            closure_params.get<std::string>("Direction"));

        if (direction == Direction::Reverse)
        {
            _direction = -1.0;
        }
    }

    // Evaluated fields
    this->addEvaluatedField(_vel_0);
    this->addEvaluatedField(_vel_1);

    this->setName("Incompressible LSVOF Vortex Velocity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFVortexVelocity<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFVortexVelocity<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _ip_coords = workset.int_rules[_ir_index]->ip_coordinates;

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFVortexVelocity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _vel_0.extent(1);

    using Kokkos::cos;
    using Kokkos::sin;

    using VertexCFD::Constants::pi;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            const double x = _ip_coords(cell, point, 0);
            const double y = _ip_coords(cell, point, 1);

            _vel_0(cell, point) = _direction * sin(2.0 * pi * y)
                                  * pow(sin(pi * x), 2.0);
            _vel_1(cell, point) = -_direction * sin(2.0 * pi * x)
                                  * pow(sin(pi * y), 2.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFVORTEXVELOCITY_IMPL_HPP
