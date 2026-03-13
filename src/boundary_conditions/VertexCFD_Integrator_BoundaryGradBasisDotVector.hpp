#ifndef VERTEXCFD_INTEGRATOR_BOUNDARYGRADBASISDOTVECTOR_HPP
#define VERTEXCFD_INTEGRATOR_BOUNDARYGRADBASISDOTVECTOR_HPP

#include <Panzer_EvaluatorStyle.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include <string>

//---------------------------------------------------------------------------//
// Special boundary integration operator for the penalty term of the boundary
// flux. Only the gradient tangential to the surface of the boundary side is
// used.
//
// NOTE: The normal vectors used by this evaluator to compute the tangential
// gradient are expected to be unit normals.
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Integrator
{
template<typename EvalType, typename Traits>
class BoundaryGradBasisDotVector
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    BoundaryGradBasisDotVector(const panzer::EvaluatorStyle& eval_style,
                               const std::string& res_name,
                               const std::string& flux_name,
                               const panzer::BasisIRLayout& basis,
                               const panzer::IntegrationRule& ir,
                               const double& multiplier = 1,
                               const std::vector<std::string>& fm_names
                               = std::vector<std::string>{});

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& fm);

    void evaluateFields(typename Traits::EvalData d);

    // Regular memory version.
    template<int NUM_FIELD_MULT>
    struct FieldMultTag
    {
    };

    template<int NUM_FIELD_MULT>
    KOKKOS_INLINE_FUNCTION void operator()(
        const FieldMultTag<NUM_FIELD_MULT>& tag,
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    // Shared memory version.
    template<int NUM_FIELD_MULT>
    struct SharedFieldMultTag
    {
    };

    template<int NUM_FIELD_MULT>
    KOKKOS_INLINE_FUNCTION void operator()(
        const SharedFieldMultTag<NUM_FIELD_MULT>& tag,
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    Teuchos::RCP<Teuchos::ParameterList> getValidParameters() const;

    enum TmpVars
    {
        REUSE,
        TMP_VAL,
        NUM_TMPS
    };

    using ScalarT = typename EvalType::ScalarT;
    using scratch_view
        = Kokkos::View<ScalarT*,
                       typename PHX::DevLayout<ScalarT>::type,
                       typename PHX::exec_space::scratch_memory_space,
                       Kokkos::MemoryUnmanaged>;

    const panzer::EvaluatorStyle _eval_style;
    double _multiplier;
    std::string _basis_name;
    std::size_t _basis_index;

    PHX::MDField<ScalarT, panzer::Cell, panzer::BASIS> _field;
    PHX::MDField<ScalarT, panzer::Cell, panzer::BASIS, panzer::Point, panzer::Dim>
        _boundary_grad_basis;

    std::vector<PHX::MDField<const ScalarT, panzer::Cell, panzer::Point>>
        _field_mults;
    Kokkos::View<Kokkos::View<const ScalarT**,
                              typename PHX::DevLayout<ScalarT>::type,
                              PHX::Device>*>
        _kokkos_field_mults;

    PHX::MDField<const ScalarT, panzer::Cell, panzer::Point, panzer::Dim> _vector;
    PHX::MDField<const ScalarT, panzer::Cell, panzer::Point, panzer::Dim> _normals;
    PHX::MDField<const double, panzer::Cell, panzer::BASIS, panzer::Point, panzer::Dim>
        _grad_basis;
};

//---------------------------------------------------------------------------//

} // end namespace Integrator
} // end namespace VertexCFD

#endif // VERTEXCFD_INTEGRATOR_BOUNDARYGRADBASISDOTVECTOR_HPP
