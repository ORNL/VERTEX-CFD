#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSLAMBDA_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSLAMBDA_HPP

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
// Lambda parameter for conservative level set method. See Quezada de Luna
// (2019)
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleCLSLambda : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    IncompressibleCLSLambda(const panzer::IntegrationRule& ir,
                            const Teuchos::ParameterList& lsvof_params,
                            const Teuchos::RCP<panzer::GlobalData>& global_data);

    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lambda;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _mag_phi;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _d_phi;

  private:
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _phi;
    PHX::MDField<const double, panzer::Cell, panzer::Point> _epsilon;

    Teuchos::RCP<panzer::GlobalData> _global_data;
    double _domain_volume;
    double _C_lambda;
    double _C_alpha;
    double _dt;
    scalar_type _avg_mag_phi;
    scalar_type _max_d_phi;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLECLSLAMBDA_HPP
