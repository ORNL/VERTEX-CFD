#ifndef VERTEXCFD_INTEGRATOR_BOUNDARYGRADBASISDOTVECTOR_IMPL_HPP
#define VERTEXCFD_INTEGRATOR_BOUNDARYGRADBASISDOTVECTOR_IMPL_HPP

#include <Panzer_BasisIRLayout.hpp>
#include <Panzer_HierarchicParallelism.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Workset_Utilities.hpp>

namespace VertexCFD
{
namespace Integrator
{
//---------------------------------------------------------------------------//
template<typename EvalType, typename Traits>
BoundaryGradBasisDotVector<EvalType, Traits>::BoundaryGradBasisDotVector(
    const panzer::EvaluatorStyle& eval_style,
    const std::string& res_name,
    const std::string& flux_name,
    const panzer::BasisIRLayout& basis,
    const panzer::IntegrationRule& ir,
    const double& multiplier,
    const std::vector<std::string>& fm_names)
    : _eval_style(eval_style)
    , _multiplier(multiplier)
    , _basis_name(basis.name())
    , _field(res_name, basis.functional)
    , _boundary_grad_basis("boundary_grad_basis", basis.basis_grad)
    , _vector(flux_name, ir.dl_vector)
    , _normals("Side Normal", ir.dl_vector)
{
    if (res_name == "")
    {
        throw std::invalid_argument(
            "Error: BoundaryGradBasisDotVector called with an empty residual "
            "name.");
    }
    if (flux_name == "")
    {
        throw std::invalid_argument(
            "Error: BoundaryGradBasisDotVector called with an empty flux "
            "name.");
    }

    if (!basis.getBasis()->supportsGrad())
    {
        const std::string msg
            = "Error:  BoundaryGradBasisDotVector:  Basis of type "
              + basis.getBasis()->name()
              + " does not support the gradient operator.";
        throw std::logic_error(msg);
    }

    if (_eval_style == panzer::EvaluatorStyle::CONTRIBUTES)
    {
        this->addContributedField(_field);
    }
    else
    {
        this->addEvaluatedField(_field);
    }

    this->addEvaluatedField(_boundary_grad_basis);

    this->addDependentField(_vector);
    this->addDependentField(_normals);

    // Add the dependent field multipliers and expand them into kokkos views
    // so we can have an arbitrary number of them on device.
    const int num_name = fm_names.size();
    _field_mults.resize(num_name);
    _kokkos_field_mults
        = Kokkos::View<Kokkos::View<const ScalarT**,
                                    typename PHX::DevLayout<ScalarT>::type,
                                    PHX::Device>*>(
            "GradBasisDotVector::KokkosFieldMultipliers", num_name);
    for (int i = 0; i < num_name; ++i)
    {
        _field_mults[i]
            = PHX::MDField<const ScalarT, panzer::Cell, panzer::Point>(
                fm_names[i], ir.dl_scalar);
        this->addDependentField(_field_mults[i]);
    }

    std::string n("BoundaryGradBasisDotVector (");
    if (_eval_style == panzer::EvaluatorStyle::CONTRIBUTES)
    {
        n += "CONTRIBUTES";
    }
    else
    {
        n += "EVALUATES";
    }
    n += "):  " + _field.fieldTag().name();
    this->setName(n);
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename Traits>
void BoundaryGradBasisDotVector<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData sd, PHX::FieldManager<Traits>& /* fm */)
{
    for (std::size_t i = 0; i < _field_mults.size(); ++i)
    {
        _kokkos_field_mults(i) = _field_mults[i].get_static_view();
    }
    PHX::Device().fence();

    _basis_index
        = panzer::getBasisIndex(_basis_name, (*sd.worksets_)[0], this->wda);
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename Traits>
template<int NUM_FIELD_MULT>
KOKKOS_INLINE_FUNCTION void
BoundaryGradBasisDotVector<EvalType, Traits>::operator()(
    const FieldMultTag<NUM_FIELD_MULT>& /* tag */,
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();

    // Create the boundary gradients.
    const int num_qp(_vector.extent(1));
    const int num_dim(_vector.extent(2));
    const int num_bases(_grad_basis.extent(1));
    const int fad_size = Kokkos::dimension_scalar(_field.get_view());
    scratch_view tmp;
    if (Sacado::IsADType<ScalarT>::value)
    {
        tmp = scratch_view(team.team_shmem(), TmpVars::NUM_TMPS, fad_size);
    }
    else
    {
        tmp = scratch_view(team.team_shmem(), TmpVars::NUM_TMPS);
    }

    auto&& n_dot_grad = tmp(TmpVars::REUSE);
    for (int qp = 0; qp < num_qp; ++qp)
    {
        for (int basis = 0; basis < num_bases; ++basis)
        {
            n_dot_grad = 0.0;
            for (int dim = 0; dim < num_dim; ++dim)
            {
                n_dot_grad += _normals(cell, qp, dim)
                              * _grad_basis(cell, basis, qp, dim);
            }
            for (int dim = 0; dim < num_dim; ++dim)
            {
                _boundary_grad_basis(cell, basis, qp, dim)
                    = _grad_basis(cell, basis, qp, dim)
                      - _normals(cell, qp, dim) * n_dot_grad;
            }
        }
    }

    // Initialize the evaluated field.
    if (_eval_style == panzer::EvaluatorStyle::EVALUATES)
    {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, 0, num_bases),
            [&](const int& basis) { _field(cell, basis) = 0.0; });
    }

    // Perform integration with the given number of field multipliers.
    auto&& tmp_value = tmp(TmpVars::TMP_VAL);
    if (NUM_FIELD_MULT == 0)
    {
        for (int qp = 0; qp < num_qp; ++qp)
        {
            for (int dim = 0; dim < num_dim; ++dim)
            {
                tmp_value = _multiplier * _vector(cell, qp, dim);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, 0, num_bases),
                    [&](const int& basis) {
                        _field(cell, basis)
                            += _boundary_grad_basis(cell, basis, qp, dim)
                               * tmp_value;
                    });
            }
        }
    }
    else if (NUM_FIELD_MULT == 1)
    {
        for (int qp = 0; qp < num_qp; ++qp)
        {
            for (int dim = 0; dim < num_dim; ++dim)
            {
                tmp_value = _multiplier * _vector(cell, qp, dim)
                            * _kokkos_field_mults(0)(cell, qp);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, 0, num_bases),
                    [&](const int& basis) {
                        _field(cell, basis)
                            += _boundary_grad_basis(cell, basis, qp, dim)
                               * tmp_value;
                    });
            }
        }
    }
    else
    {
        const int num_field_mults(_kokkos_field_mults.extent(0));
        for (int qp = 0; qp < num_qp; ++qp)
        {
            auto&& field_mults_total = tmp(TmpVars::REUSE);
            field_mults_total = 1;
            for (int fm = 0; fm < num_field_mults; ++fm)
                field_mults_total *= _kokkos_field_mults(fm)(cell, qp);
            for (int dim = 0; dim < num_dim; ++dim)
            {
                tmp_value = _multiplier * _vector(cell, qp, dim)
                            * field_mults_total;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, 0, num_bases),
                    [&](const int& basis) {
                        _field(cell, basis)
                            += _boundary_grad_basis(cell, basis, qp, dim)
                               * tmp_value;
                    });
            }
        }
    }
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename Traits>
template<int NUM_FIELD_MULT>
KOKKOS_INLINE_FUNCTION void
BoundaryGradBasisDotVector<EvalType, Traits>::operator()(
    const SharedFieldMultTag<NUM_FIELD_MULT>& /* tag */,
    const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const
{
    const int cell = team.league_rank();
    const int num_qp = _vector.extent(1);
    const int num_dim = _vector.extent(2);
    const int num_bases = _grad_basis.extent(1);
    const int fad_size = Kokkos::dimension_scalar(_field.get_view());

    scratch_view tmp;
    scratch_view tmp_field;
    if (Sacado::IsADType<ScalarT>::value)
    {
        tmp = scratch_view(team.team_shmem(), TmpVars::NUM_TMPS, fad_size);
        tmp_field = scratch_view(team.team_shmem(), num_bases, fad_size);
    }
    else
    {
        tmp = scratch_view(team.team_shmem(), TmpVars::NUM_TMPS);
        tmp_field = scratch_view(team.team_shmem(), num_bases);
    }

    // Create the boundary gradients.
    auto&& n_dot_grad = tmp(TmpVars::REUSE);
    for (int qp = 0; qp < num_qp; ++qp)
    {
        for (int basis = 0; basis < num_bases; ++basis)
        {
            n_dot_grad = 0.0;
            for (int dim = 0; dim < num_dim; ++dim)
            {
                n_dot_grad += _normals(cell, qp, dim)
                              * _grad_basis(cell, basis, qp, dim);
            }
            for (int dim = 0; dim < num_dim; ++dim)
            {
                _boundary_grad_basis(cell, basis, qp, dim)
                    = _grad_basis(cell, basis, qp, dim)
                      - _normals(cell, qp, dim) * n_dot_grad;
            }
        }
    }

    // Initialize the evaluated field.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_bases),
                         [&](const int& basis) { tmp_field(basis) = 0.0; });

    // Perform integration with the given number of fields.
    auto&& tmp_value = tmp(TmpVars::TMP_VAL);
    if (NUM_FIELD_MULT == 0)
    {
        for (int qp = 0; qp < num_qp; ++qp)
        {
            for (int dim = 0; dim < num_dim; ++dim)
            {
                team.team_barrier();
                tmp_value = _multiplier * _vector(cell, qp, dim);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, 0, num_bases),
                    [&](const int& basis) {
                        tmp_field(basis)
                            += _boundary_grad_basis(cell, basis, qp, dim)
                               * tmp_value;
                    });
            }
        }
    }
    else if (NUM_FIELD_MULT == 1)
    {
        for (int qp = 0; qp < num_qp; ++qp)
        {
            for (int dim = 0; dim < num_dim; ++dim)
            {
                team.team_barrier();
                tmp_value = _multiplier * _vector(cell, qp, dim)
                            * _kokkos_field_mults(0)(cell, qp);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, 0, num_bases),
                    [&](const int& basis) {
                        tmp_field(basis)
                            += _boundary_grad_basis(cell, basis, qp, dim)
                               * tmp_value;
                    });
            }
        }
    }
    else
    {
        const int num_field_mults(_kokkos_field_mults.extent(0));
        for (int qp = 0; qp < num_qp; ++qp)
        {
            auto&& field_mults_total = tmp(TmpVars::REUSE);
            field_mults_total = 1;
            for (int fm = 0; fm < num_field_mults; ++fm)
                field_mults_total *= _kokkos_field_mults(fm)(cell, qp);
            for (int dim = 0; dim < num_dim; ++dim)
            {
                team.team_barrier();
                tmp_value = _multiplier * _vector(cell, qp, dim)
                            * field_mults_total;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, 0, num_bases),
                    [&](const int& basis) {
                        tmp_field(basis)
                            += _boundary_grad_basis(cell, basis, qp, dim)
                               * tmp_value;
                    });
            }
        }
    }

    // Put values into target field
    if (_eval_style == panzer::EvaluatorStyle::EVALUATES)
    {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, 0, num_bases),
            [&](const int& basis) { _field(cell, basis) = tmp_field(basis); });
    }
    else if (_eval_style == panzer::EvaluatorStyle::CONTRIBUTES)
    {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, num_bases),
                             [&](const int& basis) {
                                 _field(cell, basis) += tmp_field(basis);
                             });
    }
}

//---------------------------------------------------------------------------//
template<typename EvalType, typename Traits>
void BoundaryGradBasisDotVector<EvalType, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
    _grad_basis = this->wda(workset).bases[_basis_index]->weighted_grad_basis;

    const bool use_shared_memory
        = panzer::HP::inst().useSharedMemory<ScalarT>();
    if (use_shared_memory)
    {
        int bytes;
        if (Sacado::IsADType<ScalarT>::value)
        {
            const int fad_size = Kokkos::dimension_scalar(_field.get_view());
            bytes = scratch_view::shmem_size(3, fad_size)
                    + scratch_view::shmem_size(_grad_basis.extent(1), fad_size);
        }
        else
            bytes = scratch_view::shmem_size(3)
                    + scratch_view::shmem_size(_grad_basis.extent(1));

        // The following if-block is for the sake of optimization depending on
        // the number of field multipliers.  The parallel_fors will loop over
        // the cells in the Workset and execute operator()() above.
        if (_field_mults.size() == 0)
        {
            auto policy
                = panzer::HP::inst()
                      .teamPolicy<ScalarT, SharedFieldMultTag<0>, PHX::Device>(
                          workset.num_cells)
                      .set_scratch_size(0, Kokkos::PerTeam(bytes));
            Kokkos::parallel_for(this->getName(), policy, *this);
        }
        else if (_field_mults.size() == 1)
        {
            auto policy
                = panzer::HP::inst()
                      .teamPolicy<ScalarT, SharedFieldMultTag<1>, PHX::Device>(
                          workset.num_cells)
                      .set_scratch_size(0, Kokkos::PerTeam(bytes));
            Kokkos::parallel_for(this->getName(), policy, *this);
        }
        else
        {
            auto policy
                = panzer::HP::inst()
                      .teamPolicy<ScalarT, SharedFieldMultTag<-1>, PHX::Device>(
                          workset.num_cells)
                      .set_scratch_size(0, Kokkos::PerTeam(bytes));
            Kokkos::parallel_for(this->getName(), policy, *this);
        }
    }
    else
    {
        int bytes;
        if (Sacado::IsADType<ScalarT>::value)
        {
            const int fad_size = Kokkos::dimension_scalar(_field.get_view());
            bytes = scratch_view::shmem_size(3, fad_size);
        }
        else
        {
            bytes = scratch_view::shmem_size(3);
        }

        // The following if-block is for the sake of optimization depending on
        // the number of field multipliers.  The Kokkos::parallel_fors will
        // loop over the cells in the Workset and execute operator()() above.
        if (_field_mults.size() == 0)
        {
            auto policy
                = panzer::HP::inst()
                      .teamPolicy<ScalarT, FieldMultTag<0>, PHX::Device>(
                          workset.num_cells)
                      .set_scratch_size(0, Kokkos::PerTeam(bytes));
            Kokkos::parallel_for(this->getName(), policy, *this);
        }
        else if (_field_mults.size() == 1)
        {
            auto policy
                = panzer::HP::inst()
                      .teamPolicy<ScalarT, FieldMultTag<1>, PHX::Device>(
                          workset.num_cells)
                      .set_scratch_size(0, Kokkos::PerTeam(bytes));
            Kokkos::parallel_for(this->getName(), policy, *this);
        }
        else
        {
            auto policy
                = panzer::HP::inst()
                      .teamPolicy<ScalarT, FieldMultTag<-1>, PHX::Device>(
                          workset.num_cells)
                      .set_scratch_size(0, Kokkos::PerTeam(bytes));
            Kokkos::parallel_for(this->getName(), policy, *this);
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace Integrator
} // end namespace VertexCFD

#endif // VERTEXCFD_INTEGRATOR_BOUNDARYGRADBASISDOTVECTOR_IMPL_HPP
