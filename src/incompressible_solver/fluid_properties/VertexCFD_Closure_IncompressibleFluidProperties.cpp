#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleFluidProperties.hpp"
#include "VertexCFD_Closure_IncompressibleFluidProperties_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::FluidProperties::IncompressibleFluidProperties)
