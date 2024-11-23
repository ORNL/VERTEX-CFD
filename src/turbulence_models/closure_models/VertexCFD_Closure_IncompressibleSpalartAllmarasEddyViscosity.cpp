#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleSpalartAllmarasEddyViscosity.hpp"
#include "VertexCFD_Closure_IncompressibleSpalartAllmarasEddyViscosity_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::IncompressibleSpalartAllmarasEddyViscosity)