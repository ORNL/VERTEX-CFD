#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLETAUSUPG_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLETAUSUPG_IMPL_HPP

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleTauSUPG<EvalType, Traits, NumSpaceDim>::IncompressibleTauSUPG(
    const panzer::IntegrationRule& ir,
    const Teuchos::ParameterList& fluid_params,
    const Teuchos::ParameterList& closure_params)
    : _tau_supg_cont("tau_supg_continuity", ir.dl_scalar)
    , _tau_supg_mom("tau_supg_momentum", ir.dl_scalar)
    , _tau_supg_energy("tau_supg_energy", ir.dl_scalar)
    , _element_length("element_length", ir.dl_vector)
    , _rho("density", ir.dl_scalar)
    , _nu("kinematic_viscosity", ir.dl_scalar)
    , _k("thermal_conductivity", ir.dl_scalar)
    , _cp("specific_heat_capacity", ir.dl_scalar)
    , _alpha_cont(0.5)
    , _alpha_mom(0.5)
    , _alpha_ene(0.5)
    , _solve_temp(fluid_params.get<bool>("Build Temperature Equation"))
    , _ir_degree(ir.cubature_degree)
    , _tau_model_temp(TauModelTemp::TempOneDXNodal)
{
    // Get tau model type for Navier-Stokes equations
    {
        const auto type_validator_mom = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<TauModel>(
                Teuchos::tuple<std::string>("Steady", "Transient", "NoSUPG"),
                "Transient"));

        _tau_model = type_validator_mom->getIntegralValue(
            closure_params.get<std::string>("Tau model for Navier-Stokes "
                                            "equations"));

        // SUPG coefficient
        if (closure_params.isType<double>("SUPG coefficient for continuity "
                                          "equation"))
            _alpha_cont = closure_params.get<double>(
                "SUPG coefficient for continuity equation");
        if (closure_params.isType<double>("SUPG coefficient for momentum "
                                          "equations"))
            _alpha_mom = closure_params.get<double>(
                "SUPG coefficient for momentum equations");
    }

    // Get tau model type for temperature equation
    if (_solve_temp)
    {
        const auto type_validator_temp = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<TauModelTemp>(
                Teuchos::tuple<std::string>("TempMultiDSUPGSteady",
                                            "TempMultiDSUPGTransient",
                                            "TempNoSUPG",
                                            "TempOneDXNodal",
                                            "TempOneDXUpwind"),
                "TempOneDXNodal"));
        _tau_model_temp = type_validator_temp->getIntegralValue(
            closure_params.get<std::string>("Tau model for temperature "
                                            "equation"));

        // SUPG coefficient
        if (closure_params.isType<double>("SUPG coefficient for temperature "
                                          "equation"))
            _alpha_ene = closure_params.get<double>(
                "SUPG coefficient for temperature equation");
    }

    // Evaluated fields
    this->addEvaluatedField(_tau_supg_cont);
    this->addEvaluatedField(_tau_supg_mom);
    if (_solve_temp)
        this->addEvaluatedField(_tau_supg_energy);

    // Dependent fields
    Utils::addDependentVectorField(*this, ir.dl_scalar, _velocity, "velocity_");
    this->addDependentField(_element_length);
    this->addDependentField(_nu);
    if (_solve_temp)
    {
        this->addDependentField(_rho);
        this->addDependentField(_k);
        this->addDependentField(_cp);
    }

    this->setName("Incompressible Tau SUPG " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTauSUPG<EvalType, Traits, NumSpaceDim>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTauSUPG<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    // Get FEM basis order
    _basis_order = workset.bases[_ir_index]->basis_layout->getBasis()->order();

    // Get time step
    const double dt = workset.step_size;
    _time_constant = 4.0 / (dt * dt);

    // Allocate space for thread-local temporaries in parallel for
    size_t bytes;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg_mom.get_view());
        bytes = scratch_view::shmem_size(
            _tau_supg_mom.extent(1), NUM_TMPS, fad_size);
    }
    else
    {
        bytes = scratch_view::shmem_size(_tau_supg_mom.extent(1), NUM_TMPS);
    }

    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleTauSUPG<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _tau_supg_mom.extent(1);
    using Kokkos::cosh;
    using Kokkos::sinh;
    using Kokkos::sqrt;
    const double tol = 1.0e-8;

    scratch_view tmp_field;
    if constexpr (Sacado::IsADType<scalar_type>::value)
    {
        const int fad_size = Kokkos::dimension_scalar(_tau_supg_mom.get_view());
        tmp_field
            = scratch_view(team.team_shmem(), num_point, NUM_TMPS, fad_size);
    }
    else
    {
        tmp_field = scratch_view(team.team_shmem(), num_point, NUM_TMPS);
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Initialize local variables
            auto&& u2_over_h2 = tmp_field(point, U2_OVER_H2);
            auto&& pe = tmp_field(point, PE);
            auto&& d = tmp_field(point, D);
            u2_over_h2 = 0.0;
            double h2 = 0.0;

            // Compute terms `h^2` and '\frac{u^2}{h^2}`
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                h2 += _element_length(cell, point, dim)
                      * _element_length(cell, point, dim);
                u2_over_h2 += _velocity[dim](cell, point)
                              * _velocity[dim](cell, point)
                              / (_element_length(cell, point, dim)
                                 * _element_length(cell, point, dim));
            }

            // SUPG coefficient for momentum and continuity equations
            if (_tau_model == TauModel::NoSUPG)
            {
                _tau_supg_cont(cell, point) = 0.0;
                _tau_supg_mom(cell, point) = 0.0;
            }
            else if (_tau_model == TauModel::Steady)
            {
                _tau_supg_mom(cell, point)
                    = 1.0
                      / sqrt(4.0 * u2_over_h2
                             + 9.0 * 16.0 * _nu(cell, point) * _nu(cell, point)
                                   / (h2 * h2));
            }
            else // Transient
            {
                _tau_supg_mom(cell, point)
                    = 1.0
                      / sqrt(_time_constant + 4.0 * u2_over_h2
                             + 9.0 * 16.0 * _nu(cell, point) * _nu(cell, point)
                                   / (h2 * h2));
            }
            _tau_supg_cont(cell, point) = _tau_supg_mom(cell, point);
            _tau_supg_cont(cell, point) *= _alpha_cont / _basis_order;
            _tau_supg_mom(cell, point) *= _alpha_mom / _basis_order;

            // SUPG coefficient for temperature equation
            if (_solve_temp)
            {
                if (_tau_model_temp == TauModelTemp::TempNoSUPG)
                {
                    _tau_supg_energy(cell, point) = 0.0;
                }
                else if (_tau_model_temp == TauModelTemp::TempOneDXUpwind)
                {
                    _tau_supg_energy(cell, point)
                        = _alpha_ene / _basis_order
                          * _element_length(cell, point, 0)
                          / (_velocity[0](cell, point) + tol);
                }
                else if (_tau_model_temp == TauModelTemp::TempOneDXNodal)
                {
                    // Nodal value: exact solution in 1D - second-order
                    // accuracy
                    d = _k(cell, point)
                        / (_rho(cell, point) * _cp(cell, point));
                    pe = 0.5 * _velocity[0](cell, point)
                         * _element_length(cell, point, 0) / d;
                    _tau_supg_energy(cell, point) = cosh(pe) / sinh(pe)
                                                    - 1.0 / pe;
                    _tau_supg_energy(cell, point)
                        *= _alpha_ene / _basis_order
                           * _element_length(cell, point, 0)
                           / (_velocity[0](cell, point) + tol);
                }
                else if (_tau_model_temp == TauModelTemp::TempMultiDSUPGSteady)
                {
                    d = _k(cell, point)
                        / (_rho(cell, point) * _cp(cell, point));
                    _tau_supg_energy(cell, point)
                        = _alpha_ene / _basis_order
                          / sqrt(4.0 * u2_over_h2
                                 + 9.0 * 16.0 * d * d / (h2 * h2));
                }
                else if (_tau_model_temp
                         == TauModelTemp::TempMultiDSUPGTransient)
                {
                    d = _k(cell, point)
                        / (_rho(cell, point) * _cp(cell, point));
                    _tau_supg_energy(cell, point)
                        = _alpha_ene / _basis_order
                          / sqrt(_time_constant + 4.0 * u2_over_h2
                                 + 9.0 * 16.0 * d * d / (h2 * h2));
                }
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLETAUSUPG_IMPL_HPP
