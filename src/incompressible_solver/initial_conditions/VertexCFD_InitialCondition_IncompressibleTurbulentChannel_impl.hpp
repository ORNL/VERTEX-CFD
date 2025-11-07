#ifndef VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETURBULENTCHANNEL_IMPL_HPP
#define VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETURBULENTCHANNEL_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_GlobalIndexer.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_PureBasis.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Teuchos_Array.hpp>

#include <cmath>
#include <random>
#include <string>

namespace VertexCFD
{
namespace InitialCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleTurbulentChannel<EvalType, Traits, NumSpaceDim>::
    IncompressibleTurbulentChannel(const Teuchos::ParameterList& ic_params,
                                   const bool& build_temp_equ,
                                   const panzer::PureBasis& basis)
    : _lagrange_pressure("lagrange_pressure", basis.functional)
    , _temperature("temperature", basis.functional)
    , _basis_name(basis.name())
    , _solve_temp(build_temp_equ)
    , _nu(ic_params.get<double>("Kinematic Viscosity"))
    , _h(ic_params.get<double>("Half Width"))
    , _Re_tau(ic_params.get<double>("Friction Reynolds Number"))
    , _u_tau(_Re_tau * _nu / _h)
    , _L_x(ic_params.get<double>("L_x"))
    , _L_z(ic_params.get<double>("L_z"))
    , _add_rands(true)
    , _rands(Kokkos::ViewAllocateWithoutInitializing("Turb rands"),
             basis.numCells(),
             basis.cardinality())
    , _nb_modes(3)
    , _U_0(0.0)
    , _T_init(std::numeric_limits<double>::quiet_NaN())
{
    // Get number of modes if present (default is 3 modes)
    if (ic_params.isType<int>("Number of Modes"))
    {
        _nb_modes = ic_params.get<int>("Number of Modes");
    }

    // Check to see if random perturbations should be added (default true)
    if (ic_params.isType<bool>("Add Random Perturbations"))
    {
        _add_rands = ic_params.get<bool>("Add Random Perturbations");
    }

    // Calculate bulk velocity from Re_b/Re_tau correlations
    const double Re_b = std::pow(_Re_tau / 0.09, 1.0 / 0.88);
    const double U_b = Re_b * _nu / 2.0 / _h;
    _U_0 = 1.28 * std::pow(Re_b, -0.0116) * U_b;

    // Populate random number array
    if (_add_rands)
    {
        auto rands_host
            = Kokkos::create_mirror_view(Kokkos::HostSpace{}, _rands);

        std::default_random_engine generator;
        std::uniform_real_distribution<double> interval(-1.0, 1.0);

        for (int i = 0; i < basis.numCells(); ++i)
        {
            for (int j = 0; j < basis.cardinality(); ++j)
            {
                rands_host(i, j) = interval(generator);
            }
        }

        Kokkos::deep_copy(_rands, rands_host);
    }

    // Add evaluated fields
    this->addEvaluatedField(_lagrange_pressure);
    this->addUnsharedField(_lagrange_pressure.fieldTag().clone());

    Utils::addEvaluatedVectorField(
        *this, basis.functional, _velocity, "velocity_", true);

    if (_solve_temp)
    {
        _T_init = ic_params.get<double>("Temperature");
        this->addEvaluatedField(_temperature);
        this->addUnsharedField(_temperature.fieldTag().clone());
    }

    this->setName("Turbulent Channel Initial Condition "
                  + std::to_string(num_space_dim) + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTurbulentChannel<EvalType, Traits, NumSpaceDim>::
    postRegistrationSetup(typename Traits::SetupData sd,
                          PHX::FieldManager<Traits>&)
{
    _basis_index = panzer::getPureBasisIndex(
        _basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTurbulentChannel<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    _basis_coords = this->wda(workset).bases[_basis_index]->basis_coordinates;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleTurbulentChannel<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_basis = _velocity[0].extent(1);
    const double abs_tol = 1e-10;

    using std::log;
    using std::pow;
    using std::sin;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_basis), [&](const int basis) {
            // Calculate non-dimensional wall distance
            const double y_plus
                = _u_tau
                  * (_h
                     - SmoothMath::abs(_basis_coords(cell, basis, 1), abs_tol))
                  / _nu;

            // Set linear velocity in viscous sublayer
            if (y_plus < 11.06)
            {
                _velocity[0](cell, basis) = y_plus * _u_tau;
            }
            // Set log velocity in log layer
            else
            {
                _velocity[0](cell, basis) = _u_tau
                                            * (1.0 / 0.41 * log(y_plus) + 5.2);
            }

            // Set base velocity values in y- and z-directions
            _velocity[1](cell, basis) = 0.0;
            if (num_space_dim == 3)
                _velocity[2](cell, basis) = 0.0;

            // Add random streamwise perturbations
            if (_add_rands)
            {
                _velocity[0](cell, basis) += 0.1 * _rands(cell, basis) * _U_0;
            }

            // Add Fourier modes
            for (int m = 1; m <= _nb_modes; ++m)
            {
                _velocity[0](cell, basis)
                    += 0.1
                       * (pow(_h, 2.0)
                          - pow(_basis_coords(cell, basis, 1), 2.0))
                       * sin(4.0 * m * pi * _basis_coords(cell, basis, 2)
                             / _L_z);

                if (num_space_dim == 3)
                {
                    _velocity[2](cell, basis)
                        += 0.1 * _U_0
                           * (pow(_h, 2.0)
                              - pow(_basis_coords(cell, basis, 1), 2.0))
                           * sin(4.0 * m * pi * _basis_coords(cell, basis, 0)
                                 / _L_x);
                }
            }

            // Set uniform temperature and pressure
            _lagrange_pressure(cell, basis) = 0.0;

            if (_solve_temp)
            {
                _temperature(cell, basis) = _T_init;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace InitialCondition
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_INITIALCONDITION_INCOMPRESSIBLETURBULENTCHANNEL_IMPL_HPP
