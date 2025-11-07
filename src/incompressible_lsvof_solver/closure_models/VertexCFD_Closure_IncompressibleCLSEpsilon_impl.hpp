#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSEPSILON_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSEPSILON_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"

#include <Panzer_HierarchicParallelism.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleCLSEpsilon<EvalType, Traits, NumSpaceDim>::IncompressibleCLSEpsilon(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& lsvof_params)
    : _epsilon("CLS_epsilon", ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
    , _alpha(1.5)
    , _method(EpsilonMeasurementMethod::Minimum)
{
    // Check for non-default alpha coefficient
    if (lsvof_params.isType<double>("Interface Thickness Coefficient"))
    {
        _alpha = lsvof_params.get<double>("Interface Thickness Coefficient");
    }

    // Check for measurement type (default is minimum)
    if (lsvof_params.isType<std::string>("Epsilon Measurement Method"))
    {
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<
                EpsilonMeasurementMethod>(
                Teuchos::tuple<std::string>(
                    "Minimum", "Maximum", "Magnitude", "Average"),
                "Minimum"));

        _method = type_validator->getIntegralValue(
            lsvof_params.get<std::string>("Epsilon Measurement Method"));
    }

    // Evaluated fields
    this->addEvaluatedField(_epsilon);

    // Dependent fields
    this->addDependentField(_element_length);

    this->setName("Incompressible CLS Epsilon " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleCLSEpsilon<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleCLSEpsilon<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _epsilon.extent(1);

    const double tol = 1e-10;

    using Kokkos::max;
    using Kokkos::min;
    using Kokkos::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Minimum method
            if (_method == EpsilonMeasurementMethod::Minimum)
            {
                _epsilon(cell, point) = 1e+10;

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    _epsilon(cell, point)
                        = min(_epsilon(cell, point),
                              _element_length(cell, point, dim));
                }
            }

            // Maximum method
            else if (_method == EpsilonMeasurementMethod::Maximum)
            {
                _epsilon(cell, point) = tol;

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    _epsilon(cell, point)
                        = max(_epsilon(cell, point),
                              _element_length(cell, point, dim));
                }
            }

            // Magnitude method
            else if (_method == EpsilonMeasurementMethod::Magnitude)
            {
                _epsilon(cell, point) = 0.0;

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    _epsilon(cell, point)
                        += pow(_element_length(cell, point, dim), 2.0);
                }

                _epsilon(cell, point)
                    = pow(max(_epsilon(cell, point), tol), 0.5);
            }

            // Average method
            else
            {
                _epsilon(cell, point) = 0.0;

                for (int dim = 0; dim < num_space_dim; ++dim)
                {
                    _epsilon(cell, point) += _element_length(cell, point, dim);
                }

                _epsilon(cell, point) /= num_space_dim;
            }

            _epsilon(cell, point) *= _alpha;
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSEPSILON_IMPL_HPP
