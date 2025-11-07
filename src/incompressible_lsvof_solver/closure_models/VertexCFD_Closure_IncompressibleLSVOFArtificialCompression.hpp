#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFARTIFICIALCOMPRESSION_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFARTIFICIALCOMPRESSION_HPP

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
// Multi-dimension evaluation for LSVOF artificial compression term
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFArtificialCompression
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFArtificialCompression(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& scalar_name,
        const std::string& equation_name,
        const std::string& flux_prefix = "",
        const std::string& gradient_prefix = "",
        const std::string& field_prefix = "");

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _scalar_flux;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _interface_velocity;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _scalar;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_scalar;
    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    double _C_alpha;

    enum TmpVars
    {
        MAG_U,
        MAG_GRAD_SCALAR,
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

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFARTIFICIALCOMPRESSION_HPP
