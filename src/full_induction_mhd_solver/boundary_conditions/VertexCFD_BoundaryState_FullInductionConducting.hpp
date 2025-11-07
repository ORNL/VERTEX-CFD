#ifndef VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONCONDUCTING_HPP
#define VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONCONDUCTING_HPP

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class FullInductionConducting : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    FullInductionConducting(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& bc_params,
        const MHDProperties::FullInductionMHDProperties& mhd_props);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _boundary_induced_magnetic_field;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        _boundary_scalar_magnetic_potential;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _boundary_grad_induced_magnetic_field;

  private:
    Kokkos::Array<double, num_space_dim> _bnd_magn_field;
    bool _build_magn_corr;
    bool _build_resistive_flux;
    bool _dirichlet_scalar_magn_pot;
    double _bnd_scalar_magn_pot;
    double _magnetic_permeability;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _scalar_magnetic_potential;

    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _boundary_velocity;

    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _induced_magnetic_field;

    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_induced_magnetic_field;

    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>, 3>
        _external_magnetic_field;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _resistivity;

    using scratch_view_B
        = Kokkos::View<scalar_type*,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
    // We know at compile time the size of the second dimension (num_space_dim)
    // but using a compile-time dimension triggers a bug in shmem_size when
    // scalar_type is a Sacado data type.
    using scratch_view_gradB
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_FULLINDUCTIONCONDUCTING_HPP
