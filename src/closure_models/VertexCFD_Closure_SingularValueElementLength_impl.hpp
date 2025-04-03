#ifndef VERTEXCFD_CLOSURE_SINGULARVALUEELEMENTLENGTH_IMPL_HPP
#define VERTEXCFD_CLOSURE_SINGULARVALUEELEMENTLENGTH_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_Workset_Utilities.hpp>

#include <algorithm>
#include <cmath>

/**
This class implements the logic to calculate the size of a mesh element
from the Jacobian matrix. The implementation follows the same logic as in
MFEM
(https://mfem.github.io/doxygen/html/classmfem_1_1Mesh.html#adde50e5a10f09b877c861d8371981c0c)
**/

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
SingularValueElementLength<EvalType, Traits>::SingularValueElementLength(
    const panzer::IntegrationRule& ir,
    const std::string& method,
    const std::string& prefix)
    : _element_length(prefix + "element_length", ir.dl_vector)
    , _ir_degree(ir.cubature_degree)
{
    if (method == "singular_value_min")
    {
        _method = Method::Min;
    }
    else if (method == "singular_value_max")
    {
        _method = Method::Max;
    }
    else
    {
        const std::string msg =
        "Element Length Method '" + method +
        "' is not a correct input.\n"
        "Choose between 'singular_value_min' or 'singular_value_max'";
        throw std::runtime_error(msg);
    }

    this->addEvaluatedField(_element_length);
    this->setName("Singular Value Element Length");
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void SingularValueElementLength<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>&)
{
    _ir_index = panzer::getIntegrationRuleIndex(_ir_degree, (*sd.worksets_)[0]);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void SingularValueElementLength<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _cell_jac = workset.int_rules[_ir_index]->jac;
    auto policy = panzer::HP::inst().teamPolicy<scalar_type, PHX::Device>(
        workset.num_cells);
    Kokkos::parallel_for(this->getName(), policy, *this);
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
KOKKOS_INLINE_FUNCTION void
SingularValueElementLength<EvalType, Traits>::operator()(
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_point = _element_length.extent(1);
    const int num_space_dim = _element_length.extent(2);

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 0, num_point), [&](const int point) {
            using std::sqrt;
            using std::fmin;
            using std::fmax;

            // Calculate the singular values of the jacobian matrix 'J' for
            // cell 'cell' and point 'point'. The singular values are the
            // square roots of the eigenvalues of the symmetric matrix 'J \cdot
            // J^t' that are computed by finding the roots of the quadratic
            // polynomial 'a \lambda^2 + b \lambda + c'
            const auto a11 = _cell_jac(cell, point, 0, 0);
            const auto a12 = _cell_jac(cell, point, 0, 1);
            const auto a21 = _cell_jac(cell, point, 1, 0);
            const auto a22 = _cell_jac(cell, point, 1, 1);

            // Set the coefficients 'a', 'b' and 'c'
            const double a = 1.0;
            const double b = -(a11 * a11 + a12 * a12 + a21 * a21 + a22 * a22);
            const double c = -(a11 * a21 + a12 * a22) * (a11 * a21 + a12 * a22)
                             + (a11 * a11 + a12 * a12)
                                   * (a21 * a21 + a22 * a22);

            // Compute delta value and make it is positive
            const double delta = fmax(b * b - 4 * a * c, 0.0);

            // Compute eigenvalues 'lambda1' and 'lambda2'
            const double lambda1 = 0.5 * (-b - sqrt(delta)) / a;
            const double lambda2 = 0.5 * (-b + sqrt(delta)) / a;

            // Set 'h' based on the value of 'method'
            const double h2 = _method == Method::Min ? fmin(lambda1, lambda2)
                                                     : fmax(lambda1, lambda2);
            const double h = sqrt(h2);

            // Set value of the element length (same value for all
            // directions)
            for (int i = 0; i < num_space_dim; i++)
            {
                _element_length(cell, point, i) = h;
            }
        });
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_SINGULARVALUEELEMENTLENGTH_IMPL_HPP
