#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLETAUSUPG_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLETAUSUPG_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension tau coefficient evaluation for SUPG.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleTauSUPG : public panzer::EvaluatorWithBaseImpl<Traits>,
                              public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleTauSUPG(const panzer::IntegrationRule& ir,
                          const Teuchos::ParameterList& fluid_params,
                          const Teuchos::ParameterList& closure_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _tau_supg_cont;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _tau_supg_mom;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _tau_supg_energy;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _k;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _cp;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    double _alpha_cont;
    double _alpha_mom;
    double _alpha_ene;
    bool _solve_temp;
    int _ir_degree;
    int _ir_index;
    int _basis_order;
    double _time_constant;

    enum TauModel
    {
        Steady,
        Transient,
        NoSUPG
    };
    TauModel _tau_model;

    enum TauModelTemp
    {
        TempMultiDSUPGSteady,
        TempMultiDSUPGTransient,
        TempNoSUPG,
        TempOneDXNodal,
        TempOneDXUpwind
    };
    TauModelTemp _tau_model_temp;

    enum TmpVars
    {
        U2_OVER_H2,
        PE,
        D,
        NUM_TMPS
    };

    /// View type for shared memory
    using scratch_view
        = Kokkos::View<scalar_type**,
                       typename PHX::DevLayout<scalar_type>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLETAUSUPG_HPP
