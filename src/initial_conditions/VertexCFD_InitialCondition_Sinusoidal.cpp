#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_InitialCondition_Sinusoidal.hpp"
#include "VertexCFD_InitialCondition_Sinusoidal_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::InitialCondition::Sinusoidal)
