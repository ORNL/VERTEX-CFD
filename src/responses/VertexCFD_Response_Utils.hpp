#ifndef VERTEXCFD_RESPONSE_UTILS_HPP
#define VERTEXCFD_RESPONSE_UTILS_HPP

// Need to make TEUCHOS_TEST_FOR_EXCEPTION available to Panzer.
#include <Teuchos_TestForException.hpp>

// This uses TEUCHOS_TEST_FOR_EXCEPTION, but doesn't inlude the required
// header.
#include <Panzer_WorksetDescriptor.hpp>

#include <Teuchos_ParameterList.hpp>

#include <vector>

namespace VertexCFD
{
namespace Response
{

//---------------------------------------------------------------------------//
// Build workset descriptors for side sets.
std::vector<panzer::WorksetDescriptor>
buildWorksetDescriptors(const Teuchos::ParameterList& sideset_plist);

} // namespace Response
} // namespace VertexCFD

#endif // VERTEXCFD_RESPONSE_UTILS_HPP
