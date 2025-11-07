#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Turbulent eddy viscosity for Menter's SST K-Omega turbulence model
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleSSTEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleSSTEddyViscosity(const panzer::IntegrationRule& ir,
                                   const Teuchos::ParameterList& user_params,
                                   bool on_wall_boundary = false);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _distance;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    double _a_1;
    double _beta_star;
    double _sigma_w2;
    bool _on_wall_boundary;

    enum TmpVars
    {
        TKE,
        OMEGA,
        LOG_REGION,
        NEAR_WALL,
        NUM_TMPS
    };

    /// View type for shared memory
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _sst_blending_function;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTEDDYVISCOSITY_HPP
