#ifndef VERTEXCFD_UTILS_PHASEINDEX_HPP
#define VERTEXCFD_UTILS_PHASEINDEX_HPP

#include <Phalanx_ExtentTraits.hpp>

namespace VertexCFD
{

// PHX tag denoting phase index for an MDField
struct PhaseIndex
{
};

} // namespace VertexCFD

namespace PHX
{

// Shortened version of tag for, e.g., printing DAG
template<>
std::string print<VertexCFD::PhaseIndex>();

} // namespace PHX

// Register tag as PHX extent
PHX_IS_EXTENT(VertexCFD::PhaseIndex)

#endif // VERTEXCFD_UTILS_PHASEINDEX_HPP
