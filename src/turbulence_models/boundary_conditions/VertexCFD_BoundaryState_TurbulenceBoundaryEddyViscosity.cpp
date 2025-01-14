#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_TurbulenceBoundaryEddyViscosity.hpp"
#include "VertexCFD_BoundaryState_TurbulenceBoundaryEddyViscosity_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::BoundaryCondition::TurbulenceBoundaryEddyViscosity)
