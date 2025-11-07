#ifndef VERTEXCFD_CLOSURE_SCALARTAUSUPG_HPP
#define VERTEXCFD_CLOSURE_SCALARTAUSUPG_HPP

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
// Multi-dimension tau coefficient evaluation for SUPG when applied to a scalar
// equation.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class VariableTauSUPG : public panzer::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    VariableTauSUPG(const panzer::IntegrationRule& ir,
                    const Teuchos::ParameterList& closure_params,
                    const Teuchos::ParameterList& supg_params);

    void postRegistrationSetup(typename Traits::SetupData sd,
                               PHX::FieldManager<Traits>& fm) override;

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    std::string _field_name;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _tau_supg;

  private:
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _diffusivity;

    double _alpha;
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

    enum TmpVars
    {
        U2_OVER_H2,
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

#endif // end VERTEXCFD_CLOSURE_SCALARTAUSUPG_HPP
