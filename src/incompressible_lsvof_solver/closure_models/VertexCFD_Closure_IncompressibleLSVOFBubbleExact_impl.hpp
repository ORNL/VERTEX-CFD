#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFBUBBLEEXACT_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFBUBBLEEXACT_IMPL_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFBubbleExact<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFBubbleExact(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& closure_params)
    : _lsvof_model_type(LSVOFModelType::VOF)
    , _dof_name(closure_params.get<std::string>("DOF Name"))
    , _radius(closure_params.get<double>("Bubble Radius"))
    , _ir_degree(ir.cubature_degree)
    , _exact_dof("Exact_" + _dof_name, ir.dl_scalar)
{
    // Check LSVOFModelType
    if (closure_params.isType<std::string>("LSVOF Model Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<LSVOFModelType>(
                Teuchos::tuple<std::string>("VOF", "CLS"), "VOF"));

        _lsvof_model_type = type_validator->getIntegralValue(
            closure_params.get<std::string>("LSVOF Model Type"));
    }

    // Get location from closure params
    const auto location
        = closure_params.get<Teuchos::Array<double>>("Bubble Location");

    for (int d = 0; d < num_space_dim; ++d)
    {
        _location[d] = location[d];
    }

    // Evaluated fields
    this->addEvaluatedField(_exact_dof);

    this->setName("Exact Solution LSVOF " + _dof_name + " Bubble");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFBubbleExact<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFBubbleExact<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
IncompressibleLSVOFBubbleExact<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _exact_dof.extent(1);

    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            double r = 0.0;

            for (int d = 0; d < num_space_dim; ++d)
            {
                r += pow(_location[d] - _ip_coords(cell, point, d), 2.0);
            }

            r = pow(r, 0.5);

            if (_lsvof_model_type == LSVOFModelType::VOF)
            {
                if (r <= _radius)
                {
                    _exact_dof(cell, point) = 1.0;
                }
                else
                {
                    _exact_dof(cell, point) = 0.0;
                }
            }
            else if (_lsvof_model_type == LSVOFModelType::CLS)
            {
                _exact_dof(cell, point) = _radius - r;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFBUBBLEEXACT_IMPL_HPP
