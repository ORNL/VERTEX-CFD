#ifndef VERTEXCFD_UTILS_VECTORIZEOUTPUTFIELDNAMES_HPP
#define VERTEXCFD_UTILS_VECTORIZEOUTPUTFIELDNAMES_HPP

#include <Kokkos_Core.hpp>
#include <Panzer_String_Utilities.hpp>

#include <set>

namespace VertexCFD
{
namespace VectorizeOutputFieldNames
{
inline std::vector<std::string> tokenizeParameter(
    const Teuchos::StringIndexedOrderedValueObjectContainerBase::KeyObjectPair<
        Teuchos::ParameterEntry> p)
{
    const auto& fields = Teuchos::any_cast<std::string>(p.second.getAny());
    std::vector<std::string> tokens;
    panzer::StringTokenizer(tokens, fields, ",", true);
    return tokens;
}

inline void getOutputFieldsByType(const Teuchos::ParameterList& params,
                                  std::vector<std::string>& fields)
{
    for (const auto& param : params)
    {
        const auto tokens = tokenizeParameter(param);
        fields.insert(fields.end(), tokens.begin(), tokens.end());
    }
}

inline void getOutputFields(const Teuchos::ParameterList& params,
                            std::vector<std::string>& out_fields,
                            std::vector<std::string>& out_vec_fields)
{
    // get the "Quantities" output names
    getOutputFieldsByType(params.sublist("Nodal Quantities"), out_fields);
    getOutputFieldsByType(params.sublist("Cell Quantities"), out_fields);
    getOutputFieldsByType(params.sublist("Cell Average Quantities"),
                          out_fields);
    // get rid of repeated names
    const std::set<std::string> out_set(out_fields.begin(), out_fields.end());
    out_fields.assign(out_set.begin(), out_set.end());
    // do the same for vector outputs
    getOutputFieldsByType(params.sublist("Cell Average Vectors"),
                          out_vec_fields);
    const std::set<std::string> out_vec_set(out_vec_fields.begin(),
                                            out_vec_fields.end());
    out_vec_fields.assign(out_vec_set.begin(), out_vec_set.end());
}

} // namespace VectorizeOutputFieldNames
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_VECTORIZEOUTPUTFIELDNAMES_HPP
