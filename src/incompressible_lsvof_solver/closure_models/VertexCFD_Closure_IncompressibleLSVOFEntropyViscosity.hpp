#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYVISCOSITY_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
// Multi-dimension evaluation for LSVOF entropy viscosity term
//
// NOTE: Here we are saving each term within min function of eq. 16
// so were able to plot them.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleLSVOFEntropyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleLSVOFEntropyViscosity(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& closure_params,
        const std::string& equation_name,
        const Teuchos::RCP<panzer::GlobalData>& global_data);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _artificial_viscosity;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_max;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_e;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;

    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _entropy_residual;

    Kokkos::Array<PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _velocity;

    double _cmax;
    double _ce;
    Teuchos::RCP<panzer::GlobalData> _global_data;
    scalar_type _entropy_norm;

    enum TmpVars
    {
        MAG_U,
        MIN_LENGTH,
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

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELSVOFENTROPYVISCOSITY_HPP
