#ifndef VERTEXCFD_CLOSURE_EXTERNALFIELDS_IMPL_HPP
#define VERTEXCFD_CLOSURE_EXTERNALFIELDS_IMPL_HPP

#include <Panzer_HierarchicParallelism.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
ExternalFields<EvalType, Traits>::ExternalFields(
    const std::string& evaluator_name,
    const Teuchos::RCP<const ExternalFieldsManager<Traits>>&
        external_fields_manager,
    const std::vector<std::string>& external_field_names,
    const Teuchos::RCP<const panzer::PureBasis>& basis)
    : _num_field(external_field_names.size())
    , _external_fields(_num_field)
    , _global_indexer(external_fields_manager->globalIndexer())
    , _field_ids(_num_field)
    , _ghosted_field_data(external_fields_manager->ghostedFieldData())
{
    // Setup evaluator data.
    for (int f = 0; f < _num_field; ++f)
    {
        _external_fields[f]
            = PHX::MDField<scalar_type, panzer::Cell, panzer::BASIS>(
                external_field_names[f], basis->functional);
        this->addEvaluatedField(_external_fields[f]);
    }
    this->setName(evaluator_name);

    // Get the field ids.
    for (int f = 0; f < _num_field; ++f)
    {
        _field_ids[f] = _global_indexer->getFieldNum(external_field_names[f]);
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ExternalFields<EvalType, Traits>::postRegistrationSetup(
    typename Traits::SetupData d, PHX::FieldManager<Traits>&)
{
    // Setup scratch data for reading the vector data.
    _scratch_offsets.resize(_num_field);
    const auto& workset_0 = (*d.worksets_)[0];
    auto block_id = this->wda(workset_0).block_id;

    for (int f = 0; f < _num_field; ++f)
    {
        const auto& offsets
            = _global_indexer->getGIDFieldOffsets(block_id, _field_ids[f]);
        _scratch_offsets[f] = Kokkos::View<int*, PHX::Device>(
            "external_field_offsets", offsets.size());
        auto offsets_mirror = Kokkos::create_mirror(_scratch_offsets[f]);
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            offsets_mirror(i) = offsets[i];
        }
        Kokkos::deep_copy(_scratch_offsets[f], offsets_mirror);
    }

    _scratch_lids = Kokkos::View<int**, PHX::Device>(
        "lids",
        _external_fields[0].extent(0),
        _global_indexer->getElementBlockGIDCount(block_id));
}

//---------------------------------------------------------------------------//
template<class EvalType, class Traits>
void ExternalFields<EvalType, Traits>::evaluateFields(typename Traits::EvalData d)
{
    // Get the local ids.
    _global_indexer->getElementLIDs(this->wda(d).cell_local_ids_k,
                                    _scratch_lids);

    // Extract the data.
    auto lids = _scratch_lids;
    auto field_data = _ghosted_field_data;
    for (int f = 0; f < _num_field; ++f)
    {
        auto offsets = _scratch_offsets[f];
        auto gather_field = _external_fields[f].get_static_view();
        Kokkos::parallel_for(
            Kokkos::RangePolicy<PHX::Device>(0, d.num_cells),
            KOKKOS_LAMBDA(const int cell) {
                const int num_basis = offsets.extent(0);
                for (int basis = 0; basis < num_basis; ++basis)
                {
                    auto offset = offsets(basis);
                    auto lid = lids(cell, offset);
                    gather_field(cell, basis) = field_data(lid);
                }
            });
    }
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSURE_EXTERNALFIELDS_IMPL_HPP
