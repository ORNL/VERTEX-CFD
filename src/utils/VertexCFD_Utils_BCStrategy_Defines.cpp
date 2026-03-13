#ifndef VERTEXCFD_UTILS_BCSTRATEGY_DEFINES_HPP
#define VERTEXCFD_UTILS_BCSTRATEGY_DEFINES_HPP

#include <Panzer_BC.hpp>
#include <Panzer_BCStrategy_TemplateManager.hpp>
#include <Teuchos_RCP.hpp>
#include <iostream>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
namespace Impl
{
template<template<class, int... Dims> class Strategy, int... Dims>
class TemplateBuilder
{
  public:
    TemplateBuilder(const panzer::BC& bc,
                    const Teuchos::RCP<panzer::GlobalData>& global_data)
        : _bc(bc)
        , _global_data(global_data)
    {
    }

    template<class EvalT>
    Teuchos::RCP<panzer::BCStrategyBase> build() const
    {
        Strategy<EvalT, Dims...>* ptr
            = new Strategy<EvalT, Dims...>(_bc, _global_data);
        return Teuchos::rcp(ptr);
    }

  private:
    const panzer::BC& _bc;
    const Teuchos::RCP<panzer::GlobalData> _global_data;
};
} // namespace Impl
} // namespace BoundaryCondition
} // namespace VertexCFD

#endif // VERTEXCFD_UTILS_BCSTRATEGY_DEFINES_HPP
