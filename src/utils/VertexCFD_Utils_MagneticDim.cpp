#include "VertexCFD_Utils_MagneticDim.hpp"

namespace PHX
{

// Shortened version of tag when, e.g., printing DAGs
template<>
std::string print<VertexCFD::MagneticDim>()
{
    return "Mag";
}

} // namespace PHX
