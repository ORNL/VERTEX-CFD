#ifndef VERTEXCFD_VERSION_HPP
#define VERTEXCFD_VERSION_HPP

#include "VertexCFD_Utils_config.hpp"

#include <string>

namespace VertexCFD
{
namespace Utils
{
std::string version();

std::string git_commit_hash();

} // end namespace Utils
} // end namespace VertexCFD

#endif // end VERTEXCFD_VERSION_HPP
