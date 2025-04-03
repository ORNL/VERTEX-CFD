#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_IMPL_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
TurbulenceKEpsilonWallFunction<EvalType, Traits, NumSpaceDim>::
    TurbulenceKEpsilonWallFunction(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& bc_params,
        const FluidProperties::ConstantFluidProperties& fluid_prop)
    : _boundary_k("BOUNDARY_turb_kinetic_energy", ir.dl_scalar)
    , _boundary_e("BOUNDARY_turb_dissipation_rate", ir.dl_scalar)
    , _boundary_grad_k("BOUNDARY_GRAD_turb_kinetic_energy", ir.dl_vector)
    , _boundary_grad_e("BOUNDARY_GRAD_turb_dissipation_rate", ir.dl_vector)
    , _boundary_u_tau("BOUNDARY_friction_velocity", ir.dl_scalar)
    , _boundary_y_plus("BOUNDARY_y_plus", ir.dl_scalar)
    , _wall_func_nu_t("wall_func_turbulent_eddy_viscosity", ir.dl_scalar)
    , _k("turb_kinetic_energy", ir.dl_scalar)
    , _e("turb_dissipation_rate", ir.dl_scalar)
    , _grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
    , _grad_e("GRAD_turb_dissipation_rate", ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
    , _C_mu(0.09)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _kappa(0.41)
    , _yp_tr(11.06)
    , _neumann(false)
{
    // Check for epsilon boundary specification
    if (bc_params.isType<std::string>("Epsilon Condition Type"))
    {
        const std::string bc_type
            = bc_params.get<std::string>("Epsilon Condition Type");

        if (bc_type == "Neumann")
        {
            _neumann = true;
        }
        else if (bc_type == "Dirichlet")
        {
            _neumann = false;
        }
        else
        {
            std::string msg = "Unknown Epsilon Condition Type " + bc_type;
            msg += "\nPlease choose from Dirichlet (default) or Neumann.\n";

            throw std::runtime_error(msg);
        }
    }

    // Add evaluated fields
    this->addEvaluatedField(_boundary_k);
    this->addEvaluatedField(_boundary_e);
    this->addEvaluatedField(_boundary_grad_k);
    this->addEvaluatedField(_boundary_grad_e);
    this->addEvaluatedField(_boundary_u_tau);
    this->addEvaluatedField(_boundary_y_plus);
    this->addEvaluatedField(_wall_func_nu_t);

    // Add dependent fields
    this->addDependentField(_k);
    this->addDependentField(_e);
    this->addDependentField(_grad_k);
    this->addDependentField(_grad_e);
    this->addDependentField(_normals);
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");

    this->setName("Boundary State Turbulence Model K-Epsilon Wall Function "
                  + std::to_string(_num_grad_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void TurbulenceKEpsilonWallFunction<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
TurbulenceKEpsilonWallFunction<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _grad_k.extent(1);
    const double max_tol = 1e-10;

    using std::pow;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate velocity magnitude
            scalar_type mag_u = 0.0;

            for (int d = 0; d < num_space_dim; d++)
            {
                mag_u += pow(_velocity[d](cell, point), 2.0);
            }

            mag_u = pow(SmoothMath::max(mag_u, max_tol, 0.0), 0.5);

            // Set turbulent kinetic energy boundary using Neumann condition
            _boundary_k(cell, point) = _k(cell, point);

            for (int d = 0; d < num_space_dim; ++d)
            {
                _boundary_grad_k(cell, point, d) = _grad_k(cell, point, d);

                // Subtract boundary normal component
                for (int grad_dim = 0; grad_dim < num_space_dim; ++grad_dim)
                {
                    _boundary_grad_k(cell, point, d)
                        -= _grad_k(cell, point, grad_dim)
                           * _normals(cell, point, grad_dim)
                           * _normals(cell, point, d);
                }
            }

            // Calculate friction velocity and nu_t at the wall as suggested
            // by Kuzmin et al. (2007)
            const scalar_type u_tau = SmoothMath::max(
                pow(_C_mu, 0.25)
                    * pow(SmoothMath::max(_k(cell, point), max_tol, 0.0), 0.5),
                mag_u / _yp_tr,
                0.0);

            const double nu_t_w = _kappa * _yp_tr * _nu;

            // Set epsilon and nu_t at boundary according to user setting
            if (_neumann)
            {
                // Set epsilon boundary equal to interior value
                _boundary_e(cell, point) = _e(cell, point);

                // Calculate boundary gradient
                for (int d = 0; d < num_space_dim; ++d)
                {
                    _boundary_grad_e(cell, point, d)
                        = _grad_e(cell, point, d)
                          + (_kappa * pow(u_tau, 5.0) / pow(nu_t_w, 2.0))
                                * _normals(cell, point, d);

                    // Subtract interior boundary normal component
                    for (int grad_dim = 0; grad_dim < num_space_dim; ++grad_dim)
                    {
                        _boundary_grad_e(cell, point, d)
                            -= _grad_e(cell, point, grad_dim)
                               * _normals(cell, point, grad_dim)
                               * _normals(cell, point, d);
                    }
                }

                // Set boundary nu_t to analytical solution
                _wall_func_nu_t(cell, point) = nu_t_w;
            }
            else
            {
                // Calculate epsilon boundary value
                _boundary_e(cell, point) = pow(u_tau, 4.0) / nu_t_w;

                // Set boundary gradient equal to interior gradient
                for (int d = 0; d < num_space_dim; ++d)
                {
                    _boundary_grad_e(cell, point, d) = _grad_e(cell, point, d);
                }

                // Calculate boundary nu_t with standard k-epsilon form
                _wall_func_nu_t(cell, point)
                    = _C_mu * pow(_k(cell, point), 2.0)
                      / SmoothMath::max(_e(cell, point), max_tol, 0.0);
            }

            // Fill in boundary friction velocity and y+ fields for
            // post-processing
            _boundary_u_tau(cell, point) = u_tau;
            _boundary_y_plus(cell, point) = _yp_tr;

            // TODO: define TKE production term at boundary to enforce P = D?
        });
}

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_BOUNDARYSTATE_TURBULENCEKEPSILONWALLFUNCTION_IMPL_HPP
