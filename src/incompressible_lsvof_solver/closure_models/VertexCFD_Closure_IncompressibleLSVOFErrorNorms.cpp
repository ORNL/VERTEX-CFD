#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleLSVOFErrorNorms.hpp"
#include "VertexCFD_Closure_IncompressibleLSVOFErrorNorms_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::IncompressibleLSVOFErrorNorms)
