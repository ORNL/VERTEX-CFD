#include "VertexCFD_Utils_PhaseIndex.hpp"

namespace PHX
{

// Shortened version of tag when, e.g., printing DAGs
template<>
std::string print<VertexCFD::PhaseIndex>()
{
    return "Phase";
}

} // namespace PHX
