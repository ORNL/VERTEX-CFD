#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "conduction/boundary_conditions/VertexCFD_ConductionBoundaryState_Factory.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::ConductionBoundaryStateFactory)
