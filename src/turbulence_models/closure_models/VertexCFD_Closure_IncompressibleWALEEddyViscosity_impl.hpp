#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleWALEEddyViscosity<EvalType, Traits, NumSpaceDim>::
    IncompressibleWALEEddyViscosity(const panzer::IntegrationRule& ir,
                                    const Teuchos::ParameterList& user_params)
    : _element_length("les_element_length", ir.dl_vector)
    , _C_k(0.094)
    , _C_w(0.275)
    , _k_sgs("sgs_kinetic_energy", ir.dl_scalar)
    , _nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
{
    // Check for user-defined coefficients
    if (user_params.isSublist("Turbulence Parameters"))
    {
        Teuchos::ParameterList turb_list
            = user_params.sublist("Turbulence Parameters");

        if (turb_list.isType<double>("C_w"))
        {
            _C_w = turb_list.get<double>("C_w");
        }

        if (turb_list.isType<double>("C_k"))
        {
            _C_k = turb_list.get<double>("C_k");
        }
    }

    // Add dependent fields
    this->addDependentField(_element_length);

    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    // Add evaluated fields
    this->addEvaluatedField(_k_sgs);
    this->addEvaluatedField(_nu_t);

    // Closure model name
    this->setName("Incompressible WALE Turbulent Eddy Viscosity "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleWALEEddyViscosity<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleWALEEddyViscosity<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    using std::pow;
    using std::sqrt;

    const int cell = team.league_rank();
    const int num_point = _nu_t.extent(1);
    const auto tol = 1.0e-10;
    const double one_third = 1.0 / 3.0;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate mag square of symmetric and skew symmetric
            // velocity gradient tensors, and mesh scale delta
            scalar_type mag_sqr_s = 0.0;
            scalar_type mag_sqr_w = 0.0;
            double delta = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    mag_sqr_s
                        += pow(0.5
                                   * (_grad_velocity[i](cell, point, j)
                                      + _grad_velocity[j](cell, point, i)),
                               2.0);

                    mag_sqr_w
                        += pow(0.5
                                   * (_grad_velocity[i](cell, point, j)
                                      - _grad_velocity[j](cell, point, i)),
                               2.0);
                }

                delta += pow(_element_length(cell, point, i), 2.0);
            }

            mag_sqr_s = SmoothMath::max(mag_sqr_s, tol, 0.0);
            mag_sqr_w = SmoothMath::max(mag_sqr_w, tol, 0.0);
            delta = sqrt(SmoothMath::max(delta, tol, 0.0));

            // Calculate traceless symmetric part of square of
            // velocity gradient tensor
            scalar_type mag_sqr_sd = 0.0;

            for (int i = 0; i < num_space_dim; ++i)
            {
                for (int j = 0; j < num_space_dim; ++j)
                {
                    scalar_type Sd_ij = 0.0;

                    for (int k = 0; k < num_space_dim; ++k)
                    {
                        // Symmetric terms
                        Sd_ij += 0.25
                                 * (_grad_velocity[i](cell, point, k)
                                    + _grad_velocity[k](cell, point, i))
                                 * (_grad_velocity[k](cell, point, j)
                                    + _grad_velocity[j](cell, point, k));

                        // Skew symmetric terms
                        Sd_ij += 0.25
                                 * (_grad_velocity[i](cell, point, k)
                                    - _grad_velocity[k](cell, point, i))
                                 * (_grad_velocity[k](cell, point, j)
                                    - _grad_velocity[j](cell, point, k));
                    }

                    // Subtract trace
                    if (i == j)
                    {
                        Sd_ij -= one_third * (mag_sqr_s - mag_sqr_w);
                    }

                    mag_sqr_sd += pow(Sd_ij, 2.0);
                }
            }

            mag_sqr_sd = SmoothMath::max(mag_sqr_sd, tol, 0.0);

            // Sub-grid eddy viscosity
            _nu_t(cell, point)
                = pow(_C_w * delta, 2.0) * pow(mag_sqr_sd, 3.0 / 2.0)
                  / (pow(mag_sqr_s, 5.0 / 2.0) + pow(mag_sqr_sd, 5.0 / 4.0));

            // Sub-grid kinetic energy
            _k_sgs(cell, point) = pow(_nu_t(cell, point) / (_C_k * delta), 2.0);
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEWALEEDDYVISCOSITY_IMPL_HPP
