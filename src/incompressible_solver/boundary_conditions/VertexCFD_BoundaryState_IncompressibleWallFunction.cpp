#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BoundaryState_IncompressibleWallFunction.hpp"
#include "VertexCFD_BoundaryState_IncompressibleWallFunction_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::IncompressibleWallFunction)
