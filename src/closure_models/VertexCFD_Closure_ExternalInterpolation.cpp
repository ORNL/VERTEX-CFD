#ifdef VERTEXCFD_HAVE_ARBORX

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_ExternalInterpolation.hpp"
#include "VertexCFD_Closure_ExternalInterpolation_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::ExternalInterpolation)

#endif
