#ifndef VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLELSVOFBUBBLE_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLELSVOFBUBBLE_IMPL_HPP

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <cmath>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFBubble<EvalType, Traits, NumSpaceDim>::IncompressibleLSVOFBubble(
    const Teuchos::ParameterList& ic_params, const panzer::PureBasis& basis)
    : _alpha(ic_params.get<std::string>("Phase Name"), basis.functional)
    , _phi("level_set", basis.functional)
    , _basis_name(basis.name())
    , _radius(ic_params.get<double>("Bubble Radius"))
    , _lsvof_model_type(LSVOFModelType::VOF)
    , _phase_type(PhaseType::Dispersed)
{
    // Check LSVOF model type
    if (ic_params.isType<std::string>("LSVOF Model Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<LSVOFModelType>(
                Teuchos::tuple<std::string>("VOF", "CLS"), "VOF"));

        _lsvof_model_type = type_validator->getIntegralValue(
            ic_params.get<std::string>("LSVOF Model Type"));
    }

    // Check phase type
    if (ic_params.isType<std::string>("Phase Type"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<PhaseType>(
                Teuchos::tuple<std::string>("Dispersed", "Continuous"),
                "Dispersed"));

        _phase_type = type_validator->getIntegralValue(
            ic_params.get<std::string>("Phase Type"));
    }

    // Get location from IC params
    const auto location
        = ic_params.get<Teuchos::Array<double>>("Bubble Location");

    for (int d = 0; d < num_space_dim; ++d)
    {
        _location[d] = location[d];
    }

    if (_lsvof_model_type == LSVOFModelType::VOF)
    {
        this->addEvaluatedField(_alpha);
        this->addUnsharedField(_alpha.fieldTag().clone());
    }
    else if (_lsvof_model_type == LSVOFModelType::CLS)
    {
        this->addEvaluatedField(_phi);
        this->addUnsharedField(_phi.fieldTag().clone());
    }

    this->setName("LSVOF " + ic_params.get<std::string>("Phase Name")
                  + " Bubble Initial Condition");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFBubble<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFBubble<EvalType, Traits, NumSpaceDim>::evaluateFields(
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
IncompressibleLSVOFBubble<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _lsvof_model_type == LSVOFModelType::VOF
                              ? _alpha.extent(1)
                              : _phi.extent(1);

    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            double r = 0.0;

            for (int d = 0; d < num_space_dim; ++d)
            {
                r += pow(_location[d] - _basis_coords(cell, basis, d), 2.0);
            }

            r = pow(r, 0.5);

            if (_lsvof_model_type == LSVOFModelType::VOF)
            {
                if (r <= _radius)
                {
                    if (_phase_type == PhaseType::Dispersed)
                    {
                        _alpha(cell, basis) = 1.0;
                    }
                    else
                    {
                        _alpha(cell, basis) = 0.0;
                    }
                }
                else
                {
                    if (_phase_type == PhaseType::Dispersed)
                    {
                        _alpha(cell, basis) = 0.0;
                    }
                    else
                    {
                        _alpha(cell, basis) = 1.0;
                    }
                }
            }
            else if (_lsvof_model_type == LSVOFModelType::CLS)
            {
                // CLS convention:
                // Positive when |x - c| < r for dispersed
                // Positive when |X - c| > r for continuous
                if (_phase_type == PhaseType::Dispersed)
                {
                    _phi(cell, basis) = _radius - r;
                }
                else if (_phase_type == PhaseType::Continuous)
                {
                    _phi(cell, basis) = r - _radius;
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLELSVOFBUBBLE_IMPL_HPP
