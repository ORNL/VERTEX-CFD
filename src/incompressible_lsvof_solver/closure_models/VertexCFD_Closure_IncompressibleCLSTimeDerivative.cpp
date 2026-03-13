#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleCLSTimeDerivative.hpp"
#include "VertexCFD_Closure_IncompressibleCLSTimeDerivative_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::IncompressibleCLSTimeDerivative)
