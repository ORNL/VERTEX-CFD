#ifndef VERTEXCFD_UTILS_MAGNETICDIM_HPP
#define VERTEXCFD_UTILS_MAGNETICDIM_HPP

#include <Phalanx_ExtentTraits.hpp>

namespace VertexCFD
{

// create constexpr int defining the size of the MagneticDim dimension
static constexpr int num_magnetic_field_dim = 3;

// PHX tag denoting magnetic field dimension for an MDField
struct MagneticDim
{
};

} // namespace VertexCFD

namespace PHX
{

// Shortened version of tag for, e.g., printing DAG
template<>
std::string print<VertexCFD::MagneticDim>();

} // namespace PHX

// Register tag as PHX extent
PHX_IS_EXTENT(VertexCFD::MagneticDim)

#endif // VERTEXCFD_UTILS_MAGNETICDIM_HPP
