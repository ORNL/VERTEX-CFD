#ifndef VERTEXCFD_UTILS_VECTORFIELDVIEW_HPP
#define VERTEXCFD_UTILS_VECTORFIELDVIEW_HPP

#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Phalanx_DataLayout.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace Utils
{
//---------------------------------------------------------------------------//
// This function registers an evaluated Kokkos View of vector fields indexed by
// the number of equations.
template<class ScalarType, class Traits, typename... Tags>
void addEvaluatedVectorFieldView(
    PHX::EvaluatorWithBaseImpl<Traits>& f,
    const Teuchos::RCP<PHX::DataLayout>& data_layout,
    const int num_entries,
    Kokkos::Array<PHX::MDField<ScalarType, Tags...>, Constants::MAX_NUM_VIEW>&
        kokkos_view,
    const std::string& base_name,
    const std::vector<std::string>& vec_name)
{
    for (int entry = 0; entry < num_entries; ++entry)
    {
        const std::string name = base_name + vec_name[entry];
        kokkos_view[entry]
            = PHX::MDField<ScalarType, Tags...>(name, data_layout);
        f.addEvaluatedField(kokkos_view[entry]);
    }
}

//---------------------------------------------------------------------------//
// This function registers a dependent Kokkos View of vector fields indexed by
// the number of equations.
template<class ScalarType, class Traits, typename... Tags>
void addDependentVectorFieldView(
    PHX::EvaluatorWithBaseImpl<Traits>& f,
    const Teuchos::RCP<PHX::DataLayout>& data_layout,
    const int num_entries,
    Kokkos::Array<PHX::MDField<const ScalarType, Tags...>,
                  Constants::MAX_NUM_VIEW>& kokkos_view,
    const std::string& base_name,
    const std::vector<std::string>& vec_name)
{
    for (int entry = 0; entry < num_entries; ++entry)
    {
        const std::string name = base_name + vec_name[entry];
        kokkos_view[entry]
            = PHX::MDField<const ScalarType, Tags...>(name, data_layout);
        f.addDependentField(kokkos_view[entry]);
    }
}

} // end namespace Utils
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_VECTORFIELDVIEW_HPP
