#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_BCStrategy_SolidElectricBoundaryFlux.hpp"
#include "VertexCFD_BCStrategy_SolidElectricBoundaryFlux_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_NUMSPACEDIM(
    VertexCFD::BoundaryCondition::SolidElectricBoundaryFlux)
