#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_RADFissionSourceExactSolution.hpp"
#include "VertexCFD_Closure_RADFissionSourceExactSolution_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS(
    VertexCFD::ClosureModel::RADFissionSourceExactSolution)
