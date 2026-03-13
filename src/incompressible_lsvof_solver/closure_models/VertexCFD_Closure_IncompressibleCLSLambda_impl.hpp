#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSLAMBDA_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSLAMBDA_IMPL_HPP

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
IncompressibleCLSLambda<EvalType, Traits, NumSpaceDim>::IncompressibleCLSLambda(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& lsvof_params,
    const Teuchos::RCP<panzer::GlobalData>& global_data)
    : _lambda("CLS_lambda", ir.dl_scalar)
    , _mag_phi("level_set_magnitude", ir.dl_scalar)
    , _d_phi("d_level_set", ir.dl_scalar)
    , _phi("level_set", ir.dl_scalar)
    , _epsilon("CLS_epsilon", ir.dl_scalar)
    , _global_data(global_data)
    , _domain_volume(lsvof_params.get<double>("Domain Volume"))
    , _C_lambda(10.0)
    , _C_alpha(1.5)
    , _dt(1.0)
{
    // Check for non-default regularization coefficient
    if (lsvof_params.isType<double>("Regularization Coefficient"))
    {
        _C_lambda = lsvof_params.get<double>("Regularization Coefficient");
    }

    // Check for non-default alpha coefficient
    if (lsvof_params.isType<double>("Interface Thickness Coefficient"))
    {
        _C_alpha = lsvof_params.get<double>("Interface Thickness Coefficient");
    }

    // Evaluated fields
    this->addEvaluatedField(_lambda);
    this->addEvaluatedField(_mag_phi);
    this->addEvaluatedField(_d_phi);

    // Dependent fields
    this->addDependentField(_phi);
    this->addDependentField(_epsilon);

    this->setName("Incompressible CLS Lambda " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleCLSLambda<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    using std::max;

    const auto& pl = *_global_data->pl;

    _dt = workset.step_size > 1e-10 ? workset.step_size : _dt;

    // Names for response functions
    const std::string mag_phi_int_name
        = "Level Set Magnitude - level_set_magnitude";
    const std::string d_phi_max_name = "Max Dphi - d_level_set";

    // Get average and max values
    _avg_mag_phi = pl.getValue<EvalType>(mag_phi_int_name) / _domain_volume;
    _max_d_phi = pl.getValue<EvalType>(d_phi_max_name);

    // Set max d_phi to length scale of domain when the observer reports
    // zero value (i.e. first iteration)
    if (_avg_mag_phi < 1e-8)
    {
        _max_d_phi = std::pow(_domain_volume, 1.0 / num_space_dim);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleCLSLambda<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lambda.extent(1);

    using Kokkos::abs;

    const double tol = 1e-10;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Update mag(phi) and phi - mag(phi)
            _mag_phi(cell, point) = abs(_phi(cell, point));
            _d_phi(cell, point) = abs(_mag_phi(cell, point) - _avg_mag_phi);

            // Set lambda
            _lambda(cell, point) = _epsilon(cell, point) / (_C_alpha * _dt);

            _lambda(cell, point)
                *= (_C_lambda
                    * (_epsilon(cell, point)
                       / (_C_alpha * SmoothMath::max(_max_d_phi, tol, 0.0))));
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSLAMBDA_IMPL_HPP
