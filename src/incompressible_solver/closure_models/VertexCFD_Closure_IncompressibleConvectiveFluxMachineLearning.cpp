#ifdef VERTEXCFD_HAVE_TENSORFLOW

#include "utils/VertexCFD_Utils_ExplicitTemplateInstantiation.hpp"

#include "VertexCFD_Closure_IncompressibleConvectiveFluxMachineLearning.hpp"
#include "VertexCFD_Closure_IncompressibleConvectiveFluxMachineLearning_impl.hpp"

VERTEXCFD_INSTANTIATE_TEMPLATE_CLASS_EVAL_TRAITS_NUMSPACEDIM(
    VertexCFD::ClosureModel::IncompressibleConvectiveFluxMachineLearning)

#endif // VERTEXCFD_HAVE_TENSORFLOW
