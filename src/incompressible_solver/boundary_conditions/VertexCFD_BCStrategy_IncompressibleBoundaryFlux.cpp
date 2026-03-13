#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BCStrategy_IncompressibleBoundaryFlux.hpp"
#include "VertexCFD_BCStrategy_IncompressibleBoundaryFlux_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::IncompressibleBoundaryFlux)
