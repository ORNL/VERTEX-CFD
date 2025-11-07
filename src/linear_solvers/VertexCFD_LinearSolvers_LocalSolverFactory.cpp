
#include "VertexCFD_LinearSolvers_LocalSolverFactory.hpp"

#ifdef __CUDACC__
#include "VertexCFD_LinearSolvers_CusolverGLU.hpp"
#endif
#ifdef HAVE_SUPERLUDIST
#include "VertexCFD_LinearSolvers_SuperLU.hpp"
#endif

namespace VertexCFD
{
namespace LinearSolvers
{
//---------------------------------------------------------------------------//
// Constructor
//---------------------------------------------------------------------------//
std::shared_ptr<LocalDirectSolver>
LocalSolverFactory::buildSolver(const Teuchos::ParameterList& params)
{
    // Get solver name, default to Cusolver GLU
    std::string name = "Cusolver GLU";
    if (params.isType<std::string>("Local Solver"))
        name = params.get<std::string>("Local Solver");

    if (name == "Cusolver GLU")
    {
#ifdef __CUDACC__
        return std::make_shared<CusolverGLU>(params);
#else
        throw std::runtime_error(
            "Solver option `Cusolver GLU` is not available because CUDA is "
            "not enabled.");
#endif
    }
    else if (name == "SuperLU")
#ifdef HAVE_SUPERLUDIST
        return std::make_shared<SuperLUSolver>();
#else
        throw std::runtime_error(
            "Solver option `SuperLU` is not available because SUPERLUDIST is"
            "not enabled.");
#endif
    else
    {
        const std::string msg = "Unrecognized local solver " + name;
        throw std::runtime_error(msg);
    }
}

//---------------------------------------------------------------------------//

} // namespace LinearSolvers
} // namespace VertexCFD
