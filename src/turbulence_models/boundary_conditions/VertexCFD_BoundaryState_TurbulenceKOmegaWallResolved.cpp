#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_TurbulenceKOmegaWallResolved.hpp"
#include "VertexCFD_BoundaryState_TurbulenceKOmegaWallResolved_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::BoundaryCondition::TurbulenceKOmegaWallResolved)
