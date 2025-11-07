#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BCStrategy_FullInductionMHDBoundaryFlux.hpp"
#include "VertexCFD_BCStrategy_FullInductionMHDBoundaryFlux_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::FullInductionMHDBoundaryFlux)
