#ifndef VERTEXCFD_UTILS_VECTORFIELD_HPP
#define VERTEXCFD_UTILS_VECTORFIELD_HPP

#include <Panzer_HierarchicParallelism.hpp>

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

#include <Kokkos_Array.hpp>

#include <string>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
// This function registers an evaluated Kokkos Array and also handles the
// special cases of initial conditions ('ics_data = true').
template<class ScalarType, class Traits, size_t NumSpaceDim, typename... Tags>
void addEvaluatedVectorField(
    PHX::EvaluatorWithBaseImpl<Traits>& f,
    const Teuchos::RCP<PHX::DataLayout>& data_layout,
    Kokkos::Array<PHX::MDField<ScalarType, Tags...>, NumSpaceDim>& kokkos_array,
    const std::string& scalar_name,
    const bool unshared = false)
{
    constexpr size_t num_space_dim = NumSpaceDim;

    for (size_t dim = 0; dim < num_space_dim; ++dim)
    {
        const std::string name = scalar_name + std::to_string(dim);
        kokkos_array[dim]
            = PHX::MDField<ScalarType, Tags...>(name, data_layout);
        f.addEvaluatedField(kokkos_array[dim]);
        if (unshared)
            f.addUnsharedField(kokkos_array[dim].fieldTag().clone());
    }
}

//---------------------------------------------------------------------------//
// This function registers a contributed Kokkos Array
template<class ScalarType, class Traits, size_t NumSpaceDim, typename... Tags>
void addContributedVectorField(
    PHX::EvaluatorWithBaseImpl<Traits>& f,
    const Teuchos::RCP<PHX::DataLayout>& data_layout,
    Kokkos::Array<PHX::MDField<ScalarType, Tags...>, NumSpaceDim>& kokkos_array,
    const std::string& scalar_name)
{
    constexpr size_t num_space_dim = NumSpaceDim;

    for (size_t dim = 0; dim < num_space_dim; ++dim)
    {
        const std::string name = scalar_name + std::to_string(dim);
        kokkos_array[dim]
            = PHX::MDField<ScalarType, Tags...>(name, data_layout);
        f.addContributedField(kokkos_array[dim]);
    }
}

//---------------------------------------------------------------------------//
// This function registers a dependent Kokkos Array
template<class ScalarType, class Traits, size_t NumSpaceDim, typename... Tags>
void addDependentVectorField(
    PHX::EvaluatorWithBaseImpl<Traits>& f,
    const Teuchos::RCP<PHX::DataLayout>& data_layout,
    Kokkos::Array<PHX::MDField<const ScalarType, Tags...>, NumSpaceDim>&
        kokkos_array,
    const std::string& scalar_name)
{
    constexpr size_t num_space_dim = NumSpaceDim;

    for (size_t dim = 0; dim < num_space_dim; ++dim)
    {
        const std::string name = scalar_name + std::to_string(dim);
        kokkos_array[dim]
            = PHX::MDField<const ScalarType, Tags...>(name, data_layout);
        f.addDependentField(kokkos_array[dim]);
    }
}

//---------------------------------------------------------------------------//

} // end namespace Utils
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_VECTORFIELD_HPP
