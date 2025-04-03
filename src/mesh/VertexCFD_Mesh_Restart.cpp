#include "VertexCFD_Mesh_Restart.hpp"

#include <Epetra_Vector.h>
#include <Panzer_NodeType.hpp>
#include <Thyra_DefaultSpmdVector.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Tpetra_MultiVector.hpp>

#include <Phalanx_config.hpp>

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace VertexCFD
{
namespace Mesh
{
//---------------------------------------------------------------------------//
// Base
//---------------------------------------------------------------------------//
MPI_Datatype Restart::setupDofMapData(
    const Teuchos::RCP<const panzer_stk::STK_Interface>& mesh,
    const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
    int& dofmap_offset,
    int& local_num_own_elem,
    std::vector<int>& owned_element_lids)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        dof_manager->getComm());
    MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*comm);
    const int comm_rank = comm->getRank();
    const int comm_size = comm->getSize();

    // Compute the maximum number of dofs per element.
    std::vector<std::string> block_ids;
    dof_manager->getElementBlockIds(block_ids);
    int max_num_dof = 0;
    for (auto& block : block_ids)
    {
        max_num_dof = std::max(max_num_dof,
                               dof_manager->getElementBlockGIDCount(block));
    }
    dofmap_offset = max_num_dof + 1;

    // Create an ordered list of the elements that we own.
    std::vector<stk::mesh::Entity> owned_elements;
    mesh->getMyElements(owned_elements);
    local_num_own_elem = owned_elements.size();
    std::vector<uint64_t> owned_element_gids(local_num_own_elem);
    for (int i = 0; i < local_num_own_elem; ++i)
    {
        owned_element_gids[i] = mesh->elementGlobalId(owned_elements[i]);
    }
    std::sort(owned_element_gids.begin(), owned_element_gids.end());

    // On rank zero determine how many elements each rank owns.
    std::vector<int> elems_per_rank;
    if (0 == comm_rank)
    {
        elems_per_rank.resize(comm_size);
    }
    MPI_Gather(&local_num_own_elem,
               1,
               MPI_INT,
               elems_per_rank.data(),
               1,
               MPI_INT,
               0,
               mpi_comm);

    // Gather the global ids to rank zero.
    const uint64_t global_num_elem
        = mesh->getEntityCounts(stk::topology::ELEM_RANK);
    std::vector<uint64_t> all_element_ids;
    std::vector<int> rank_displacements;
    if (0 == comm_rank)
    {
        all_element_ids.resize(global_num_elem);
        rank_displacements.resize(comm_size);
        rank_displacements[0] = 0;
        std::partial_sum(elems_per_rank.begin(),
                         elems_per_rank.end() - 1,
                         rank_displacements.begin() + 1);
    }
    MPI_Gatherv(owned_element_gids.data(),
                owned_element_gids.size(),
                MPI_UINT64_T,
                all_element_ids.data(),
                elems_per_rank.data(),
                rank_displacements.data(),
                MPI_UINT64_T,
                0,
                mpi_comm);

    // Sort the list of global ids on rank zero and determine the sorted
    // offset of each element into the global dof map.
    if (0 == comm_rank)
    {
        std::vector<uint64_t> sorted_index(global_num_elem);
        std::iota(sorted_index.begin(), sorted_index.end(), 0);
        std::sort(sorted_index.begin(),
                  sorted_index.end(),
                  [&](uint64_t i, uint64_t j) {
                      return all_element_ids[i] < all_element_ids[j];
                  });
        for (uint64_t i = 0; i < global_num_elem; ++i)
        {
            all_element_ids[sorted_index[i]] = i;
        }
    }

    // Scatter the sorted gids back to the owners so they know how to write to
    // their part of the dofmap.
    std::vector<uint64_t> sorted_elem_positions(local_num_own_elem);
    MPI_Scatterv(all_element_ids.data(),
                 elems_per_rank.data(),
                 rank_displacements.data(),
                 MPI_UINT64_T,
                 sorted_elem_positions.data(),
                 sorted_elem_positions.size(),
                 MPI_UINT64_T,
                 0,
                 mpi_comm);

    // Create the local element ids in the sorted order and the displacments.
    owned_element_lids.resize(local_num_own_elem);
    std::vector<int> displacements(local_num_own_elem);
    for (int i = 0; i < local_num_own_elem; ++i)
    {
        owned_element_lids[i] = mesh->elementLocalId(owned_element_gids[i]);
        displacements[i] = dofmap_offset * sorted_elem_positions[i];
    }

    // Create the MPI datatype for the dof map.
    // Make indexed data types into which we will write the local data.
    MPI_Datatype indexed;
    MPI_Type_create_indexed_block(displacements.size(),
                                  dofmap_offset,
                                  displacements.data(),
                                  MPI_UINT64_T,
                                  &indexed);

    // Update the extent of the datatype. This new data type will need to be
    // committed.
    MPI_Aint extent = global_num_elem * dofmap_offset * sizeof(uint64_t);
    MPI_Datatype dofmap_type;
    MPI_Type_create_resized(indexed, 0, extent, &dofmap_type);
    MPI_Type_free(&indexed);
    return dofmap_type;
}

//---------------------------------------------------------------------------//
// Writer
// NOTE: allow_dofmap_overwrite is false by default and should be provided only
//       in unit tests.
//---------------------------------------------------------------------------//
RestartWriter::RestartWriter(
    const Teuchos::RCP<const panzer_stk::STK_Interface>& mesh,
    const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
    const Teuchos::ParameterList& output_params,
    const bool allow_dofmap_overwrite)
    : _dof_manager(dof_manager)
    , _file_prefix(output_params.get<std::string>("Restart File Prefix"))
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        dof_manager->getComm());
    MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*comm);
    const int comm_rank = comm->getRank();

    // Setup dof map data.
    int dofmap_offset;
    int local_num_own_elem;
    std::vector<int> owned_element_lids;
    MPI_Datatype dofmap_type = this->setupDofMapData(mesh,
                                                     dof_manager,
                                                     dofmap_offset,
                                                     local_num_own_elem,
                                                     owned_element_lids);
    MPI_Type_commit(&dofmap_type);

    // Map global element id to global dof ids.
    std::vector<panzer::GlobalOrdinal> element_dofs;
    std::vector<uint64_t> dofmap(local_num_own_elem * dofmap_offset);
    int elem_local_offset;
    int elem_num_dof;
    for (int i = 0; i < local_num_own_elem; ++i)
    {
        // Get the element dofs.
        _dof_manager->getElementGIDs(owned_element_lids[i], element_dofs);
        elem_num_dof = element_dofs.size();

        // Map DOF ids to the STK mesh element id.
        elem_local_offset = dofmap_offset * i;
        dofmap[elem_local_offset] = elem_num_dof;
        for (int d = 0; d < elem_num_dof; ++d)
        {
            dofmap[elem_local_offset + d + 1] = element_dofs[d];
        }

        // Fill the unused dof map spots with a known quantity.
        for (int d = elem_num_dof; d < dofmap_offset - 1; ++d)
        {
            dofmap[elem_local_offset + d + 1]
                = std::numeric_limits<uint64_t>::max();
        }
    }

    // Open a binary data file for the dof map.
    const std::string dofmap_file_name = _file_prefix + ".restart.dofmap";
    MPI_File dofmap_file;
    {
        int file_mode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
        if (!allow_dofmap_overwrite)
            file_mode |= MPI_MODE_EXCL;

        const int error_code = MPI_File_open(mpi_comm,
                                             dofmap_file_name.c_str(),
                                             file_mode,
                                             MPI_INFO_NULL,
                                             &dofmap_file);
        if (MPI_SUCCESS != error_code)
        {
            char error_string[MPI_MAX_ERROR_STRING + 1]{};
            int error_string_len = 0;
            MPI_Error_string(error_code, error_string, &error_string_len);
            error_string[error_string_len] = 0;

            std::string msg{"Error creating DOF Map file `" + dofmap_file_name
                            + "' : "};
            msg += error_string;

            throw std::runtime_error(msg);
        }
    }

    // Write the dof map offset into the header on rank 0.
    MPI_File_set_view(
        dofmap_file, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_Offset header_size;
    if (0 == comm_rank)
    {
        MPI_File_write(
            dofmap_file, &dofmap_offset, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_get_position(dofmap_file, &header_size);
        MPI_File_get_byte_offset(dofmap_file, header_size, &header_size);
    }

    // Broadcast the header size.
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, mpi_comm);

    // Write dof map to file.
    MPI_File_set_view(dofmap_file,
                      header_size,
                      MPI_UINT64_T,
                      dofmap_type,
                      "native",
                      MPI_INFO_NULL);
    MPI_File_write_all(dofmap_file,
                       dofmap.data(),
                       dofmap.size(),
                       MPI_UINT64_T,
                       MPI_STATUS_IGNORE);

    // Cleanup.
    MPI_File_close(&dofmap_file);
    MPI_Type_free(&dofmap_type);

    // Make indexed data types into which we will write the local data.
    std::vector<panzer::GlobalOrdinal> gids;
    _dof_manager->getOwnedIndices(gids);
    _displacements.resize(gids.size());
    std::copy(gids.begin(), gids.end(), _displacements.begin());

    // Construct map from owned global ids back to local ordering
    for (int i = 0; i < _dof_manager->getNumOwned(); ++i)
        _global_to_local.insert({gids[i], i});

    // Sort displacements
    std::sort(_displacements.begin(), _displacements.end());
    MPI_Datatype indexed;
    MPI_Type_create_indexed_block(
        _displacements.size(), 1, _displacements.data(), MPI_DOUBLE, &indexed);

    // Update the extent of the datatype.
    uint64_t global_size = gids.size();
    MPI_Allreduce(
        MPI_IN_PLACE, &global_size, 1, MPI_UINT64_T, MPI_SUM, mpi_comm);
    MPI_Aint extent = global_size * sizeof(double);
    MPI_Type_create_resized(indexed, 0, extent, &_dof_type);
    MPI_Type_free(&indexed);
    MPI_Type_commit(&_dof_type);
}

//---------------------------------------------------------------------------//
RestartWriter::~RestartWriter()
{
    MPI_Type_free(&_dof_type);
}

//---------------------------------------------------------------------------//
void RestartWriter::writeSolution(
    const Teuchos::RCP<const Thyra::VectorBase<double>>& x,
    const Teuchos::RCP<const Thyra::VectorBase<double>>& x_dot,
    const int index,
    const double time)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        _dof_manager->getComm());
    MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*comm);
    const int comm_rank = comm->getRank();

    // SpmdVectorBase is a common base class of either Epetra or Tpetra
    // implementations that has a concept of local vs. global portions
    // of a vector.
    auto x_spmd
        = Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(x);
    auto x_dot_spmd
        = Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<double>>(x_dot);
    auto spmd_space = x_spmd->spmdSpace();
    const int local_size = spmd_space->localSubDim();
    const int global_size = spmd_space->dim();

    // Check parallel data.
    if (local_size != _dof_manager->getNumOwned())
    {
        throw std::logic_error(
            "Thyra::VectorBase and panzer::GlobalIndexer local sizes do not "
            "match");
    }

    // Open a binary data file.
    std::stringstream file_name;
    file_name << _file_prefix << "_" << index;
    std::string restart_file_name = file_name.str() + ".restart.data";
    MPI_File restart_file;
    MPI_File_open(mpi_comm,
                  restart_file_name.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE,
                  MPI_INFO_NULL,
                  &restart_file);

    // Write the time into the header on rank 0.
    const int num_fields = _dof_manager->getNumFields();
    MPI_File_set_view(
        restart_file, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_Offset header_size;
    if (0 == comm_rank)
    {
        MPI_File_write(restart_file, &time, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_write(
            restart_file, &num_fields, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_write(
            restart_file, &global_size, 1, MPI_INT, MPI_STATUS_IGNORE);
        MPI_File_get_position(restart_file, &header_size);
        MPI_File_get_byte_offset(restart_file, header_size, &header_size);
    }

    // Broadcast the header size.
    MPI_Bcast(&header_size, 1, MPI_OFFSET, 0, mpi_comm);

    // Reorder the state vector to correspond to the increasing order
    // displacements.
    auto x_view = x_spmd->getLocalSubVector();
    std::vector<double> x_copy(local_size);
    for (int i = 0; i < local_size; ++i)
    {
        const int local_index = _global_to_local[_displacements[i]];
        x_copy[i] = x_view(local_index);
    }

    // Write state vector.
    MPI_File_set_view(restart_file,
                      header_size,
                      MPI_DOUBLE,
                      _dof_type,
                      "native",
                      MPI_INFO_NULL);
    MPI_File_write_all(restart_file,
                       x_copy.data(),
                       x_copy.size(),
                       MPI_DOUBLE,
                       MPI_STATUS_IGNORE);

    // Reorder the state vector time derivative to correspond to the
    // increasing order displacements.
    auto x_dot_view = x_dot_spmd->getLocalSubVector();
    std::vector<double> x_dot_copy(local_size);
    for (int i = 0; i < local_size; ++i)
    {
        const int local_index = _global_to_local[_displacements[i]];
        x_dot_copy[i] = x_dot_view(local_index);
    }

    // Write state vector time derivative.
    MPI_File_write_all(restart_file,
                       x_dot_copy.data(),
                       x_dot_copy.size(),
                       MPI_DOUBLE,
                       MPI_STATUS_IGNORE);

    // Cleanup.
    MPI_File_close(&restart_file);
}

//---------------------------------------------------------------------------//
// Reader
//---------------------------------------------------------------------------//
RestartReader::RestartReader(const Teuchos::RCP<const Teuchos::Comm<int>>& comm,
                             const Teuchos::ParameterList& input_params)
    : _restart_file_name(input_params.get<std::string>("Restart Data File "
                                                       "Name"))
    , _dofmap_file_name(input_params.get<std::string>("Restart DOF Map File "
                                                      "Name"))
{
    // Get the MPI communicator.
    MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*comm);

    // Open the restart file.
    MPI_File restart_file;
    const int error_code = MPI_File_open(mpi_comm,
                                         _restart_file_name.c_str(),
                                         MPI_MODE_RDONLY,
                                         MPI_INFO_NULL,
                                         &restart_file);

    if (MPI_SUCCESS != error_code)
    {
        std::string msg = "\n\nThe restart file " + _restart_file_name
                          + "\ncould not be found in the working directory.\n";
        throw std::logic_error(msg);
    }

    // Get the initial state time.
    MPI_File_set_view(
        restart_file, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_File_read_all(restart_file, &_t_init, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);

    // Cleanup.
    MPI_File_close(&restart_file);
}

//---------------------------------------------------------------------------//
void RestartReader::readSolution(
    const Teuchos::RCP<const panzer_stk::STK_Interface>& mesh,
    const Teuchos::RCP<const panzer::GlobalIndexer>& dof_manager,
    const Teuchos::RCP<Thyra::VectorBase<double>>& x,
    const Teuchos::RCP<Thyra::VectorBase<double>>& x_dot)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        dof_manager->getComm());
    MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*comm);

    // SpmdVectorBase is a common base class of either Epetra or Tpetra
    // implementations that has a concept of local vs. global portions
    // of a vector.
    auto x_spmd = Teuchos::rcp_dynamic_cast<Thyra::SpmdVectorBase<double>>(x);
    auto x_dot_spmd
        = Teuchos::rcp_dynamic_cast<Thyra::SpmdVectorBase<double>>(x_dot);
    auto spmd_space = x_spmd->spmdSpace();
    const int local_size = spmd_space->localSubDim();
    const int global_size = spmd_space->dim();

    // Check parallel data.
    if (local_size != dof_manager->getNumOwned())
    {
        throw std::logic_error(
            "Thyra::VectorBase and panzer::GlobalIndexer local sizes do not "
            "match");
    }

    // Setup dof map data.
    int dofmap_offset;
    int local_num_own_elem;
    std::vector<int> owned_element_lids;
    MPI_Datatype dofmap_type = this->setupDofMapData(mesh,
                                                     dof_manager,
                                                     dofmap_offset,
                                                     local_num_own_elem,
                                                     owned_element_lids);
    MPI_Type_commit(&dofmap_type);

    // Open the dof map file.
    MPI_File dofmap_file;
    const int error_code = MPI_File_open(mpi_comm,
                                         _dofmap_file_name.c_str(),
                                         MPI_MODE_RDONLY,
                                         MPI_INFO_NULL,
                                         &dofmap_file);

    if (MPI_SUCCESS != error_code)
    {
        std::string msg = "\n\nThe DOFMAP file " + _dofmap_file_name
                          + "\ncould not be found in the working directory.\n";
        throw std::logic_error(msg);
    }

    // Read the dof map offset from the header on rank 0.
    int file_dofmap_offset;
    MPI_Offset dofmap_header_size;
    MPI_File_set_view(
        dofmap_file, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_File_read_all(
        dofmap_file, &file_dofmap_offset, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_get_position(dofmap_file, &dofmap_header_size);
    MPI_File_get_byte_offset(
        dofmap_file, dofmap_header_size, &dofmap_header_size);
    if (file_dofmap_offset != dofmap_offset)
    {
        throw std::logic_error("DOF map offsets do not match");
    }

    // Read the local dof map.
    std::vector<uint64_t> dofmap(dofmap_offset * local_num_own_elem);
    MPI_File_set_view(dofmap_file,
                      dofmap_header_size,
                      MPI_UINT64_T,
                      dofmap_type,
                      "native",
                      MPI_INFO_NULL);
    MPI_File_read_all(dofmap_file,
                      dofmap.data(),
                      dofmap.size(),
                      MPI_UINT64_T,
                      MPI_STATUS_IGNORE);

    // Cleanup dof map file.
    MPI_File_close(&dofmap_file);
    MPI_Type_free(&dofmap_type);

    // Map the local dof map to the new dof manager.
    std::unordered_map<panzer::GlobalOrdinal, int> mapped_gids;
    std::unordered_map<int, panzer::GlobalOrdinal> reverse_mapped_gids;
    std::vector<panzer::GlobalOrdinal> element_dofs;
    for (int i = 0; i < local_num_own_elem; ++i)
    {
        // Get the element dofs.
        dof_manager->getElementGIDs(owned_element_lids[i], element_dofs);
        const int elem_num_dof = dofmap[dofmap_offset * i];
        if (static_cast<unsigned>(elem_num_dof) != element_dofs.size())
        {
            throw std::logic_error(
                "DOF map and DOF manager element sizes do not match");
        }

        // Map to the dofs in the restart file.
        for (int d = 0; d < elem_num_dof; ++d)
        {
            mapped_gids.emplace(element_dofs[d],
                                dofmap[dofmap_offset * i + d + 1]);
            reverse_mapped_gids.emplace(dofmap[dofmap_offset * i + d + 1],
                                        element_dofs[d]);
        }
    }

    // Open the restart file.
    MPI_File restart_file;
    MPI_File_open(mpi_comm,
                  _restart_file_name.c_str(),
                  MPI_MODE_RDONLY,
                  MPI_INFO_NULL,
                  &restart_file);

    // Get the header data.
    MPI_Offset restart_header_size;
    double time;
    int num_fields;
    int global_num_dof;
    MPI_File_set_view(
        restart_file, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
    MPI_File_read_all(restart_file, &time, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read_all(restart_file, &num_fields, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_read_all(
        restart_file, &global_num_dof, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_get_position(restart_file, &restart_header_size);
    MPI_File_get_byte_offset(
        restart_file, restart_header_size, &restart_header_size);
    if (time != _t_init)
    {
        throw std::logic_error("Restart initialization times do not match");
    }
    if (num_fields != dof_manager->getNumFields())
    {
        throw std::logic_error("Restart number of fields do not match");
    }
    if (global_num_dof != global_size)
    {
        throw std::logic_error("Restart global number of DOFs do not match");
    }

    // Make indexed data types from which we will read the local data.
    std::vector<panzer::GlobalOrdinal> owned_gids;
    dof_manager->getOwnedIndices(owned_gids);
    std::vector<int> restart_displacements(local_size);
    for (int i = 0; i < local_size; ++i)
    {
        restart_displacements[i] = mapped_gids.find(owned_gids[i])->second;
    }

    // Create map from owned global ids back to local ids
    std::unordered_map<panzer::GlobalOrdinal, int> global_to_local;
    for (int i = 0; i < local_size; ++i)
        global_to_local.insert({owned_gids[i], i});

    std::sort(restart_displacements.begin(), restart_displacements.end());
    MPI_Datatype restart_indexed;
    MPI_Type_create_indexed_block(local_size,
                                  1,
                                  restart_displacements.data(),
                                  MPI_DOUBLE,
                                  &restart_indexed);

    // Update the extent of the dof datatype.
    MPI_Aint extent = global_size * sizeof(double);
    MPI_Datatype dof_type;
    MPI_Type_create_resized(restart_indexed, 0, extent, &dof_type);
    MPI_Type_free(&restart_indexed);
    MPI_Type_commit(&dof_type);

    // Read state vector.
    std::vector<double> x_copy(local_size);
    MPI_File_set_view(restart_file,
                      restart_header_size,
                      MPI_DOUBLE,
                      dof_type,
                      "native",
                      MPI_INFO_NULL);
    MPI_File_read_all(restart_file,
                      x_copy.data(),
                      x_copy.size(),
                      MPI_DOUBLE,
                      MPI_STATUS_IGNORE);

    // Reorder the state vector to correspond to the increasing order
    // displacements.
    panzer::GlobalOrdinal new_gid;
    int new_lid;
    auto x_view = x_spmd->getNonconstLocalSubVector();
    std::vector<double> reordered_data(local_size);
    for (int i = 0; i < local_size; ++i)
    {
        new_gid = reverse_mapped_gids.find(restart_displacements[i])->second;
        new_lid = global_to_local[new_gid];
        reordered_data[new_lid] = x_copy[i];
    }
    this->update_vector(x, reordered_data);

    // Read state vector time derivative.
    std::vector<double> x_dot_copy(local_size);
    MPI_File_read_all(restart_file,
                      x_dot_copy.data(),
                      x_dot_copy.size(),
                      MPI_DOUBLE,
                      MPI_STATUS_IGNORE);

    // Reorder the state vector time derivative to correspond to the
    // increasing order displacements.
    for (int i = 0; i < local_size; ++i)
    {
        new_gid = reverse_mapped_gids.find(restart_displacements[i])->second;
        new_lid = global_to_local[new_gid];
        reordered_data[new_lid] = x_dot_copy[i];
    }
    this->update_vector(x_dot, reordered_data);

    // Cleanup
    MPI_File_close(&restart_file);
    MPI_Type_free(&dof_type);
}

//---------------------------------------------------------------------------//
void RestartReader::update_vector(
    const Teuchos::RCP<Thyra::VectorBase<double>>& vec,
    const std::vector<double>& values) const
{
    // Vector should be either a DefaultSpmdVector (Epetra) or a TpetraVector
    auto spmd_vec
        = Teuchos::rcp_dynamic_cast<Thyra::DefaultSpmdVector<double>>(vec);
    auto thyratpetra_vec = Teuchos::rcp_dynamic_cast<
        Thyra::TpetraVector<double, int, panzer::GlobalOrdinal, panzer::TpetraNodeType>>(
        vec);

    if (spmd_vec == Teuchos::null && thyratpetra_vec == Teuchos::null)
        throw std::runtime_error("Unrecognized Thyra vector type");

    if (spmd_vec != Teuchos::null)
    {
        auto space = vec->space();
        auto epetra_map = Thyra::get_Epetra_Map(space);
        auto epetra_vec = Thyra::get_Epetra_Vector(vec, epetra_map);
        std::copy(values.begin(), values.end(), epetra_vec->Values());
    }
    else
    {
        auto tpetra_vec = thyratpetra_vec->getTpetraVector();
        auto data_view = tpetra_vec->getLocalViewHost(
            Tpetra::Access::OverwriteAllStruct());
        std::copy(values.begin(), values.end(), data_view.data());
    }
}

//---------------------------------------------------------------------------//

} // end namespace Mesh
} // end namespace VertexCFD
