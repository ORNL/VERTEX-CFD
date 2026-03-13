#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFUNCTION_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFUNCTION_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLSVOFEntropyFunction<EvalType, Traits, NumSpaceDim>::
    IncompressibleLSVOFEntropyFunction(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& scalar_name,
        const std::string& equation_name)
    : _entropy_function("entropy_function", ir.dl_scalar)
    , _entropy_residual("entropy_residual", ir.dl_scalar)
    , _grad_scalar("GRAD_" + scalar_name, ir.dl_vector)
    , _scalar(scalar_name, ir.dl_scalar)
    , _dxdt_scalar("DXDT_" + scalar_name, ir.dl_scalar)
    , _entropy_type(EntropyTypes::quadratic)
    , _factor_entropy(closure_params.get<double>("Time Derivative Entropy "
                                                 "Scaling"))
{
    // Get entropy type
    const auto type_validator = Teuchos::rcp(
        new Teuchos::StringToIntegralParameterEntryValidator<EntropyTypes>(
            Teuchos::tuple<std::string>("Quadratic", "Log", "Biquadratic"),
            "Quadratic"));
    _entropy_type = type_validator->getIntegralValue(
        closure_params.get<std::string>("Entropy Type"));

    // Evaluated fields
    this->addEvaluatedField(_entropy_function);
    this->addEvaluatedField(_entropy_residual);

    // Dependent fields
    this->addDependentField(_grad_scalar);
    this->addDependentField(_scalar);
    this->addDependentField(_dxdt_scalar);

    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    this->setName(equation_name + " Incompressible LSVOF Entropy Viscosity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLSVOFEntropyFunction<EvalType, Traits, NumSpaceDim>::
    evaluateFields(typename Traits::EvalData workset)
{
    // Allocate space for thread-local temporaries
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_grad_scalar.get_view());
        bytes = scratch_view::shmem_size(
            _grad_scalar.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_grad_scalar.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLSVOFEntropyFunction<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using Kokkos::log;
    using Kokkos::pow;
    using SmoothMath::max;
    using SmoothMath::min;
    const int cell = team.league_rank();
    const int num_point = _grad_scalar.extent(1);
    const double max_tol = 1e-10;

    scratch_view scratch_data;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_grad_scalar.get_view());

        scratch_data
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        scratch_data = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            auto&& tmp_scalar = scratch_data(point, TMP_SCALAR);
            auto&& tmp_log_scalar = scratch_data(point, TMP_LOG_SCALAR);

            _entropy_residual(cell, point) = 0.0;
            for (int d = 0; d < num_space_dim; ++d)
            {
                _entropy_residual(cell, point)
                    += _velocity[d](cell, point) * _grad_scalar(cell, point, d);
            }

            if (_entropy_type == EntropyTypes::quadratic)
            {
                _entropy_residual(cell, point) += _dxdt_scalar(cell, point)
                                                  * _factor_entropy;
                _entropy_residual(cell, point) *= _scalar(cell, point);

                _entropy_function(cell, point)
                    = 0.5 * pow(_scalar(cell, point), 2.0);
            }
            else if (_entropy_type == EntropyTypes::log)
            {
                tmp_scalar = min(max(_scalar(cell, point), max_tol, 0.0),
                                 1.0 - max_tol,
                                 0.0);
                tmp_log_scalar = log(tmp_scalar) - log(1.0 - tmp_scalar);

                _entropy_residual(cell, point) *= tmp_log_scalar;

                _entropy_residual(cell, point)
                    += log(tmp_scalar / (1.0 - tmp_scalar))
                       * _dxdt_scalar(cell, point) * _factor_entropy;

                _entropy_function(cell, point)
                    = (tmp_scalar * log(tmp_scalar))
                      + ((1.0 - tmp_scalar) * log(1.0 - tmp_scalar));
            }
            else if (_entropy_type == EntropyTypes::biquadratic)
            {
                _entropy_residual(cell, point)
                    *= 2.0 * _scalar(cell, point)
                       - 4.0 * pow(_scalar(cell, point), 3.0);
                _entropy_residual(cell, point)
                    += 2.0 * _scalar(cell, point)
                       * (1.0 - 2.0 * pow(_scalar(cell, point), 2.0))
                       * _dxdt_scalar(cell, point) * _factor_entropy;

                _entropy_function(cell, point)
                    = pow(_scalar(cell, point), 2.0)
                      - pow(_scalar(cell, point), 4.0);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYFUNCTION_IMPL_HPP
