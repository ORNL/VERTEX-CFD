#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSEPSILON_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSEPSILON_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalData.hpp>

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
// Interface thickness for conservative level set method. See Quezada de Luna
// (2019)
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleCLSEpsilon : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleCLSEpsilon(const panzer::IntegrationRule& ir,
                             const Teuchos::ParameterList& lsvof_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<double, panzer::Cell, panzer::Point> _epsilon;

  private:
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim>
        _element_length;

    double _alpha;

    enum EpsilonMeasurementMethod
    {
        Minimum,
        Maximum,
        Magnitude,
        Average
    };

    EpsilonMeasurementMethod _method;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSEPSILON_HPP
