#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_FullInductionClosureModelFactory.hpp"
#include "VertexCFD_FullInductionClosureModelFactory_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::ClosureModel::FullInductionFactory)
