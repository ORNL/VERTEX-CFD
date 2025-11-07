#ifndef VERTEXCFD_UTILS_SCALARTOVECTOR_IMPL_HPP
#define VERTEXCFD_UTILS_SCALARTOVECTOR_IMPL_HPP

#include <Phalanx_DataLayout_MDALayout.hpp>

#include <Kokkos_Core.hpp>

#include <limits>
#include <utility>

namespace VertexCFD
{
namespace Utils
{

//---------------------------------------------------------------------------//
template<typename EvalType, typename NewTag>
ScalarToVector<EvalType, NewTag>::ScalarToVector(
    const panzer::IntegrationRule& ir,
    const std::string& vector_name,
    const VecOpt<std::string>& scalar_names,
    const bool time_deriv,
    const bool grads)
    : _vector_name{vector_name}
    , _scalar_names{scalar_names}
{
    addFields("", _vector_field, _scalar_fields, ir.dl_scalar);

    if (grads)
    {
        addFields(
            "GRAD_", _vector_grad_field, _scalar_grad_fields, ir.dl_vector);
    }

    if (time_deriv)
    {
        addFields(
            "DXDT_", _vector_dxdt_field, _scalar_dxdt_fields, ir.dl_scalar);
    }

    this->setName("ScalarToVector (" + vector_name + ")");
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename NewTag>
void ScalarToVector<EvalType, NewTag>::evaluateFields(
    typename panzer::Traits::EvalData)
{
    copyFields(_vector_field, _scalar_fields);
    copyFields(_vector_grad_field, _scalar_grad_fields);
    copyFields(_vector_dxdt_field, _scalar_dxdt_fields);
}

//---------------------------------------------------------------------------//
namespace Impl
{
// Make a new Layout by, taking all but the last extent from an existing
// layout.
template<typename... Tags, std::size_t... Is>
auto makeLayout(const PHX::DataLayout& layout,
                std::index_sequence<Is...>,
                const int num_scalars)
{
    static_assert(sizeof...(Is) + 1 == sizeof...(Tags));

    return Teuchos::rcp(
        new PHX::MDALayout<Tags...>(layout.extent(Is)..., num_scalars));
}
} // namespace Impl

// Add dependent and evaluated fields for scalars and vector, respectively.
template<typename EvalType, typename NewTag>
template<typename... ScalarTags>
void ScalarToVector<EvalType, NewTag>::addFields(
    const char* prefix,
    PHX::MDField<ScalarT, ScalarTags..., NewTag>& vector_field,
    VecOpt<PHX::MDField<const ScalarT, ScalarTags...>>& scalar_fields,
    const Teuchos::RCP<PHX::DataLayout>& scalar_layout)
{
    const int num_scalars = _scalar_names.size();

    scalar_fields.reserve(num_scalars);
    for (const auto& name : _scalar_names)
    {
        if (name)
            scalar_fields.emplace_back(
                std::in_place, prefix + *name, scalar_layout);
        else
            scalar_fields.emplace_back(std::nullopt);
    }

    auto vector_layout = Impl::makeLayout<ScalarTags..., NewTag>(
        *scalar_layout, std::index_sequence_for<ScalarTags...>{}, num_scalars);

    vector_field = PHX::MDField<ScalarT, ScalarTags..., NewTag>(
        prefix + _vector_name, vector_layout);

    this->addEvaluatedField(vector_field);
    for (const auto& field : scalar_fields)
    {
        if (field)
            this->addDependentField(*field);
    }
}

//---------------------------------------------------------------------------//
namespace Impl
{
// Make a dependent ALL so that it can be duplicated through pack expansion.
template<typename>
constexpr auto ALL = Kokkos::ALL;
} // namespace Impl

// Perform copies from scalars to vector.
template<typename EvalType, typename NewTag>
template<typename... ScalarTags>
void ScalarToVector<EvalType, NewTag>::copyFields(
    PHX::MDField<ScalarT, ScalarTags..., NewTag>& vector_field,
    VecOpt<PHX::MDField<const ScalarT, ScalarTags...>>& scalar_fields)
{
    const int num_scalars = scalar_fields.size();

    for (int sc = 0; sc < num_scalars; ++sc)
    {
        auto vector_view = Kokkos::subview(
            vector_field.get_view(), Impl::ALL<ScalarTags>..., sc);
        if (scalar_fields[sc])
        {
            Kokkos::deep_copy(vector_view, scalar_fields[sc]->get_view());
        }
        else
        {
            Kokkos::deep_copy(vector_view,
                              std::numeric_limits<double>::signaling_NaN());
        }
    }
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename NewTag>
auto ScalarToVector<EvalType, NewTag>::createFromIndexed(
    const panzer::IntegrationRule& ir,
    const std::string& vector_name,
    const int num_scalars,
    const bool time_deriv,
    const bool grads)
{
    auto scalar_names = VecOpt<std::string>{};
    scalar_names.reserve(num_scalars);
    for (int n = 0; n < num_scalars; ++n)
    {
        const auto ns = std::to_string(n);
        scalar_names.emplace_back(vector_name + "_" + ns);
    }
    return Teuchos::rcp(
        new ScalarToVector(ir, vector_name, scalar_names, time_deriv, grads));
};

//---------------------------------------------------------------------------//
template<typename EvalType, typename NewTag>
auto ScalarToVector<EvalType, NewTag>::createFromSuffixed(
    const panzer::IntegrationRule& ir,
    const std::string& vector_name,
    const VecOpt<std::string>& scalar_suffixes,
    const bool time_deriv,
    const bool grads)
{
    auto scalar_names = VecOpt<std::string>{};
    scalar_names.reserve(scalar_suffixes.size());
    for (const auto& suffix : scalar_suffixes)
    {
        std::optional<std::string> name;
        if (suffix)
            name = vector_name + "_" + *suffix;
        scalar_names.emplace_back(std::move(name));
    }
    return Teuchos::rcp(
        new ScalarToVector(ir, vector_name, scalar_names, time_deriv, grads));
};

//---------------------------------------------------------------------------//
template<typename EvalType, typename NewTag>
auto ScalarToVector<EvalType, NewTag>::createFromList(
    const panzer::IntegrationRule& ir,
    const std::string& vector_name,
    const std::vector<std::string>& scalar_list,
    const bool time_deriv,
    const bool grads)
{
    auto scalar_names = VecOpt<std::string>{};
    scalar_names.reserve(scalar_list.size());
    for (const auto& name : scalar_list)
    {
        scalar_names.emplace_back(std::move(name));
    }
    return Teuchos::rcp(
        new ScalarToVector(ir, vector_name, scalar_names, time_deriv, grads));
};

//---------------------------------------------------------------------------//

} // namespace Utils
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_SCALARTOVECTOR_IMPL_HPP
