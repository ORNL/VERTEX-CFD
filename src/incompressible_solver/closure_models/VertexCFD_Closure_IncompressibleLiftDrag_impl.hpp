#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLELIFTDRAG_IMPL_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLELIFTDRAG_IMPL_HPP

#include "utils/VertexCFD_Utils_SmoothMath.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
IncompressibleLiftDrag<EvalType, Traits, NumSpaceDim>::IncompressibleLiftDrag(
    const panzer::IntegrationRule& ir,
    const FluidProperties::ConstantFluidProperties& fluid_prop,
    const Teuchos::ParameterList& user_params)
    : _normals("Side Normal", ir.dl_vector)
    , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
    , _nu(fluid_prop.constantKinematicViscosity())
    , _rho(fluid_prop.constantDensity())
    , _use_compressible_formula(user_params.isType<bool>("Compressible "
                                                         "Formula")
                                    ? user_params.get<bool>("Compressible "
                                                            "Formula")
                                    : false)
{
    // Add evaluated field
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _shear_tensor, "shear_tensor_");
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _viscous_force, "viscous_force_");
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _pressure_force, "pressure_force_");
    Utils::addEvaluatedVectorField(
        *this, ir.dl_scalar, _total_force, "total_force_");

    // Add dependent field
    this->addDependentField(_normals);
    this->addDependentField(_lagrange_pressure);
    Utils::addDependentVectorField(
        *this, ir.dl_vector, _grad_velocity, "GRAD_velocity_");

    this->setName("Incompressible Lift/Drag " + std::to_string(num_space_dim)
                  + "D");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
void IncompressibleLiftDrag<EvalType, Traits, NumSpaceDim>::evaluateFields(
    typename Traits::EvalData workset)
{
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
KOKKOS_INLINE_FUNCTION void
IncompressibleLiftDrag<EvalType, Traits, NumSpaceDim>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _lagrange_pressure.extent(1);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            // Calculate wall shear tensor (grad.U + transpose(grad.U))
            for (int i = 0; i < num_space_dim; ++i)
            {
                _shear_tensor[i](cell, point) = 0;
                for (int j = 0; j < num_space_dim; ++j)
                {
                    _shear_tensor[i](cell, point)
                        += (_grad_velocity[j](cell, point, i)
                            + _grad_velocity[i](cell, point, j))
                           * _normals(cell, point, j);
                    // If compressible formula div.U != 0 hence calculate
                    // deviatoric part
                    if (_use_compressible_formula)
                    {
                        _shear_tensor[i](cell, point)
                            -= 2.0 * _grad_velocity[j](cell, point, j)
                               / num_space_dim * _normals(cell, point, i);
                    }
                }
                _pressure_force[i](cell, point)
                    = _lagrange_pressure(cell, point)
                      * _normals(cell, point, i);
                _viscous_force[i](cell, point)
                    = -_rho * _nu * _shear_tensor[i](cell, point);
                _total_force[i](cell, point)
                    = _viscous_force[i](cell, point)
                      + _pressure_force[i](cell, point);
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_INCOMPRESSIBLELIFTDRAG_IMPL_HPP
