#ifndef VERTEXCFD_CLOSURE_INDUCTIONRESISTIVEFLUX_IMPL_HPP
#define VERTEXCFD_CLOSURE_INDUCTIONRESISTIVEFLUX_IMPL_HPP

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
InductionResistiveFlux<EvalType, Traits, NumSpaceDim>::InductionResistiveFlux(
    const panzer::IntegrationRule& ir,
    const MHDProperties::FullInductionMHDProperties& mhd_props,
    const std::string& flux_prefix,
    const std::string& gradient_prefix)
    : _magnetic_correction_potential_flux(
          flux_prefix + "RESISTIVE_FLUX_magnetic_correction_potential",
          ir.dl_vector)
    , _resistivity("resistivity", ir.dl_scalar)
    , _grad_resistivity("GRAD_resistivity", ir.dl_vector)
    , _variable_resistivity(mhd_props.variableResistivity())
    , _solve_magn_corr(mhd_props.buildMagnCorr())
    , _magnetic_permeability(mhd_props.vacuumMagneticPermeability())
{
    // Evaluated fields
    this->addEvaluatedField(_magnetic_correction_potential_flux);

    Utils::addEvaluatedVectorField(*this, ir.dl_vector, _induction_flux,
                                 flux_prefix + "RESISTIVE_FLUX_"
                                               "induction_");

    if (_solve_magn_corr)
    {
        this->addEvaluatedField(_magnetic_correction_potential_flux);
    }

    // Dependent fields
    this->addDependentField(_resistivity);
    if (_variable_resistivity)
    {
        this->addDependentField(_grad_resistivity);
    }

    Utils::addDependentVectorField(
        *this, ir.dl_scalar, _total_magnetic_field, "total_magnetic_field_");
    Utils::addDependentVectorField(
        *this,
        ir.dl_vector,
        _grad_total_magnetic_field,
        gradient_prefix + "GRAD_total_magnetic_field_");

    this->setName("Induction Resistive Flux " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void InductionResistiveFlux<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
InductionResistiveFlux<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _induction_flux[0].extent(1);
    const int num_grad_dim = _induction_flux[0].extent(2);
    const double mu_0_inv = 1.0 / _magnetic_permeability;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // for constant resistivity Div(eta B) = eta*Div(B)
            scalar_type div_eta_b = 0.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                div_eta_b += _grad_total_magnetic_field[dim](cell, point, dim);
            }
            div_eta_b *= _resistivity(cell, point);

            // eta * Grad(B) contribution
            for (int flux_dim = 0; flux_dim < num_grad_dim; ++flux_dim)
            {
                for (int vec_dim = 0; vec_dim < num_space_dim; ++vec_dim)
                {
                    _induction_flux[vec_dim](cell, point, flux_dim)
                        = _resistivity(cell, point)
                          * _grad_total_magnetic_field[vec_dim](
                              cell, point, flux_dim);
                }
            }

            if (_variable_resistivity)
            {
                // Div(eta B) = eta*Div(B) + grad(eta).B
                for (int dim = 0; dim < num_grad_dim; ++dim)
                {
                    div_eta_b += _grad_resistivity(cell, point, dim)
                                 * _total_magnetic_field[dim](cell, point);
                }

                // B \otimes grad(eta) contribution
                for (int flux_dim = 0; flux_dim < num_grad_dim; ++flux_dim)
                {
                    for (int vec_dim = 0; vec_dim < num_grad_dim; ++vec_dim)
                    {
                        _induction_flux[vec_dim](cell, point, flux_dim)
                            += _total_magnetic_field[flux_dim](cell, point)
                               * _grad_resistivity(cell, point, vec_dim);
                    }
                }
            }

            // -Div(eta B) * I contribution
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                _induction_flux[dim](cell, point, dim) -= div_eta_b;
            }

            // every term has eta or grad(eta), so can scale once by mu_0 to
            // recover \hat(eta)
            for (int flux_dim = 0; flux_dim < num_grad_dim; ++flux_dim)
            {
                for (int vec_dim = 0; vec_dim < num_space_dim; ++vec_dim)
                {
                    _induction_flux[vec_dim](cell, point, flux_dim) *= mu_0_inv;
                }
            }

            if (_solve_magn_corr)
            {
                for (int flux_dim = 0; flux_dim < num_grad_dim; ++flux_dim)
                {
                    _magnetic_correction_potential_flux(cell, point, flux_dim)
                        = 0.0;
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INDUCTIONRESISTIVEFLUX_IMPL_HPP
