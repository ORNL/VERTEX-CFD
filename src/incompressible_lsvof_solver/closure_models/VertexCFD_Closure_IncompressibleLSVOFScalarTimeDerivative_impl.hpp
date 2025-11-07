#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARTIMEDERIVATIVE_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARTIMEDERIVATIVE_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleLSVOFScalarTimeDerivative<EvalType, Traits>::
    IncompressibleLSVOFScalarTimeDerivative(const panzer::IntegrationRule& ir,
                                            const std::string& dof_name,
                                            const std::string& eqn_name,
                                            const bool& mass_weighted)
    : _dqdt_dof("DQDT_" + eqn_name, ir.dl_scalar)
    , _dof(dof_name, ir.dl_scalar)
    , _rho("density", ir.dl_scalar)
    , _dxdt_dof("DXDT_" + dof_name, ir.dl_scalar)
    , _dxdt_rho("DXDT_density", ir.dl_scalar)
    , _mass_weighted(mass_weighted)
{
    // Evaluated fields
    this->addEvaluatedField(_dqdt_dof);

    // Dependent fields
    this->addDependentField(_dxdt_dof);

    if (_mass_weighted)
    {
        this->addDependentField(_dof);
        this->addDependentField(_rho);
        this->addDependentField(_dxdt_rho);
    }

    this->setName("Incompressible LSVOF " + dof_name + " Time Derivative");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleLSVOFScalarTimeDerivative<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFScalarTimeDerivative<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _dqdt_dof.extent(1);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            if (_mass_weighted)
            {
                _dqdt_dof(cell, point)
                    = _rho(cell, point) * _dxdt_dof(cell, point)
                      + _dxdt_rho(cell, point) * _dof(cell, point);
            }
            else
            {
                _dqdt_dof(cell, point) = _dxdt_dof(cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFSCALARTIMEDERIVATIVE_IMPL_HPP
