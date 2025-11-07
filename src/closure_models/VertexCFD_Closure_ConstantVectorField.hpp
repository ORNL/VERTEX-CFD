#ifndef VERTEXCFD_CLOSURE_CONSTANTVECTORFIELD_HPP
#define VERTEXCFD_CLOSURE_CONSTANTVECTORFIELD_HPP

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
// Constant Vector Field.
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class ConstantVectorField : public panzer::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    ConstantVectorField(const panzer::IntegrationRule& ir,
                        const Teuchos::ParameterList& closure_params);

    void evaluateFields(typename Traits::EvalData workset) override;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _vector_field;

  private:
    Kokkos::Array<double, num_space_dim> _vector;
    int _ir_degree;

    std::string _vec_field_name;
};

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_CONSTANTVECTORFIELD_HPP
