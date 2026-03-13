#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSSIGN_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSSIGN_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
IncompressibleCLSSign<EvalType, Traits>::IncompressibleCLSSign(
    const panzer::IntegrationRule& ir,
    const std::string& field_prefix,
    const bool& eval_star_sign)
    : _heaviside(field_prefix + "CLS_heaviside", ir.dl_scalar)
    , _sign(field_prefix + "CLS_sign", ir.dl_scalar)
    , _sign_star(field_prefix + "STAR_CLS_sign", ir.dl_scalar)
    , _phi(field_prefix + "level_set", ir.dl_scalar)
    , _phi_star(field_prefix + "STAR_level_set", ir.dl_scalar)
    , _epsilon("CLS_epsilon", ir.dl_scalar)
    , _eval_star_sign(eval_star_sign)
{
    // Evaluated fields
    this->addEvaluatedField(_heaviside);
    this->addEvaluatedField(_sign);
    if (_eval_star_sign)
        this->addEvaluatedField(_sign_star);

    // Dependent fields
    this->addDependentField(_phi);
    if (_eval_star_sign)
        this->addDependentField(_phi_star);
    this->addDependentField(_epsilon);

    this->setName("Incompressible CLS Sign");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void IncompressibleCLSSign<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void IncompressibleCLSSign<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _sign.extent(1);

    using Kokkos::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Update current sign field
            if (_phi(cell, point) < (-_epsilon(cell, point)))
            {
                _heaviside(cell, point) = 0.0;
            }
            else if (_phi(cell, point) > _epsilon(cell, point))
            {
                _heaviside(cell, point) = 1.0;
            }
            else
            {
                _heaviside(cell, point)
                    = 0.5
                      * (1.0 + _phi(cell, point) / _epsilon(cell, point)
                         + 1.0 / pi
                               * sin(pi * _phi(cell, point)
                                     / _epsilon(cell, point)));
            }

            _sign(cell, point) = 2.0 * _heaviside(cell, point) - 1.0;

            // Update star sign field
            if (_eval_star_sign)
            {
                if (_phi_star(cell, point) < (-_epsilon(cell, point)))
                {
                    _sign_star(cell, point) = 0.0;
                }
                else if (_phi_star(cell, point) > _epsilon(cell, point))
                {
                    _sign_star(cell, point) = 1.0;
                }
                else
                {
                    _sign_star(cell, point)
                        = 0.5
                          * (1.0
                             + _phi_star(cell, point) / _epsilon(cell, point)
                             + 1.0 / pi
                                   * sin(pi * _phi_star(cell, point)
                                         / _epsilon(cell, point)));
                }

                _sign_star(cell, point) = 2.0 * _sign_star(cell, point) - 1.0;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSSIGN_IMPL_HPP
