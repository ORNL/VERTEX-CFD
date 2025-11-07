#ifndef VERTEXCFD_UTILS_SCALARTOVECTOR_HPP
#define VERTEXCFD_UTILS_SCALARTOVECTOR_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Traits.hpp>

#include <Teuchos_RCP.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include <optional>
#include <string>
#include <vector>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
template<typename EvalType, typename NewTag>
class ScalarToVector : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                       public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
  public:
    // Create a vector field, vector_name, from a set of scalar fields,
    // vector_name_0, vector_name_1, ...
    static auto createFromIndexed(const panzer::IntegrationRule& ir,
                                  const std::string& vector_name,
                                  const int num_scalars,
                                  const bool time_deriv,
                                  const bool grads);

    // Create a vector field, vector_name, from a set of scalar fields,
    // vector_name_(scalar_suffixes[0]), vector_name_(scalar_suffixes[1]), ...
    //
    // If an (optional) suffix is unset, the respective entry in the vector
    // field will be initialized to NaN and must be set elsewhere.
    static auto createFromSuffixed(
        const panzer::IntegrationRule& ir,
        const std::string& vector_name,
        const std::vector<std::optional<std::string>>& scalar_suffixes,
        const bool time_deriv,
        const bool grads);

    // Create a vector field, vector_name, from a list of scalar field names
    static auto createFromList(const panzer::IntegrationRule& ir,
                               const std::string& vector_name,
                               const std::vector<std::string>& num_scalars,
                               const bool time_deriv,
                               const bool grads);

    void evaluateFields(typename panzer::Traits::EvalData) override;

  private:
    template<typename T>
    using VecOpt = std::vector<std::optional<T>>;

    // Constuct the evaluator
    ScalarToVector(const panzer::IntegrationRule& ir,
                   const std::string& vector_name,
                   const VecOpt<std::string>& scalar_names,
                   const bool time_deriv,
                   const bool grads);

    using ScalarT = typename EvalType::ScalarT;

    // Helper to add fields
    template<typename... ScalarTags>
    void
    addFields(const char* prefix,
              PHX::MDField<ScalarT, ScalarTags..., NewTag>& vector_field,
              VecOpt<PHX::MDField<const ScalarT, ScalarTags...>>& scalar_fields,
              const Teuchos::RCP<PHX::DataLayout>& scalar_layout);

    // Helper to evaluate/copy fields
    template<typename... ScalarTags>
    void copyFields(
        PHX::MDField<ScalarT, ScalarTags..., NewTag>& vector_field,
        VecOpt<PHX::MDField<const ScalarT, ScalarTags...>>& scalar_fields);

    const std::string _vector_name;
    const VecOpt<std::string> _scalar_names;

    // Dependent scalar fields
    VecOpt<PHX::MDField<const ScalarT, panzer::Cell, panzer::Point>> _scalar_fields;
    VecOpt<PHX::MDField<const ScalarT, panzer::Cell, panzer::Point>>
        _scalar_dxdt_fields;
    VecOpt<PHX::MDField<const ScalarT, panzer::Cell, panzer::Point, panzer::Dim>>
        _scalar_grad_fields;

  public:
    // Evaluated vector fields
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point, NewTag> _vector_field;
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point, NewTag> _vector_dxdt_field;
    PHX::MDField<ScalarT, panzer::Cell, panzer::Point, panzer::Dim, NewTag>
        _vector_grad_field;
};
//---------------------------------------------------------------------------//

} // namespace Utils
} // namespace VertexCFD

#include "VertexCFD_Utils_ScalarToVector_impl.hpp"

#endif // VERTEXCFD_UTILS_SCALARTOVECTOR_HPP
