#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BCStrategy_ConductionBoundaryFlux.hpp"
#include "VertexCFD_BCStrategy_ConductionBoundaryFlux_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::ConductionBoundaryFlux)
