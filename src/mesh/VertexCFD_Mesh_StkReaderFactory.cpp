#include "VertexCFD_Mesh_StkReaderFactory.hpp"

#include <PanzerAdaptersSTK_config.hpp>
#include <Panzer_STK_Interface.hpp>

#include <Ionit_Initializer.h>
#include <Ioss_Decomposition.h>
#include <Ioss_ElementBlock.h>
#include <Ioss_Region.h>
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

namespace VertexCFD
{
namespace Mesh
{
//---------------------------------------------------------------------------//
int getMeshDimension(const std::string& mesh_str,
                     stk::ParallelMachine parallel_mach,
                     const bool is_exodus)
{
    stk::io::StkMeshIoBroker mesh_data(parallel_mach);
    mesh_data.property_add(Ioss::Property("LOWER_CASE_VARIABLE_NAMES", false));
    if (is_exodus)
        mesh_data.add_mesh_database(mesh_str, "exodusII", stk::io::READ_MESH);
    else
        mesh_data.add_mesh_database(mesh_str, "cgns", stk::io::READ_MESH);
    mesh_data.create_input_mesh();
    return Teuchos::as<int>(mesh_data.meta_data().spatial_dimension());
}

//---------------------------------------------------------------------------//
StkReaderFactory::StkReaderFactory()
    : file_name_("")
    , decomp_method_("RIB")
    , restart_index_(0)
    , is_exodus_(true)
    , user_mesh_scaling_(false)
    , mesh_scale_factor_(0.0)
    , levels_of_refinement_(0)
{
}

//---------------------------------------------------------------------------//
StkReaderFactory::StkReaderFactory(const std::string& file_name,
                                   const int restart_index,
                                   const bool is_exodus)
    : file_name_(file_name)
    , decomp_method_("RIB")
    , restart_index_(restart_index)
    , is_exodus_(is_exodus)
    , user_mesh_scaling_(false)
    , mesh_scale_factor_(0.0)
    , levels_of_refinement_(0)
{
}

//---------------------------------------------------------------------------//
Teuchos::RCP<panzer_stk::STK_Interface>
StkReaderFactory::buildMesh(stk::ParallelMachine parallel_mach) const
{
    auto mesh = buildUncommitedMesh(parallel_mach);

    const bool buildRefinementSupport = levels_of_refinement_ > 0 ? true
                                                                  : false;
    mesh->initialize(parallel_mach, false, buildRefinementSupport);

    completeMeshConstruction(*mesh, parallel_mach);

    return mesh;
}

//---------------------------------------------------------------------------//
/** This builds all the meta data of the mesh. Does not call metaData->commit.
 * Allows user to add solution fields and other pieces. The mesh can be
 * "completed" by calling <code>completeMeshConstruction</code>.
 */
Teuchos::RCP<panzer_stk::STK_Interface>
StkReaderFactory::buildUncommitedMesh(stk::ParallelMachine parallel_mach) const
{
    // read in meta data
    stk::io::StkMeshIoBroker* mesh_data
        = new stk::io::StkMeshIoBroker(parallel_mach);

    mesh_data->use_simple_fields();

    // Tell IOSS to keep variable names as is; do not convert to lower case.
    mesh_data->property_add(Ioss::Property("LOWER_CASE_VARIABLE_NAMES", false));

    // Tell IOSS not to combine scalar fields into vector or tensor fields
    // based on suffix recognition. We need scalar fields in order to use them
    // as initial conditions.
    mesh_data->property_add(Ioss::Property("ENABLE_FIELD_RECOGNITION", false));

    mesh_data->property_add(
        Ioss::Property("DECOMPOSITION_METHOD", decomp_method_));

    // add in "FAMILY_TREE" entity for doing refinement
    std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
    entity_rank_names.push_back("FAMILY_TREE");
    mesh_data->set_rank_name_vector(entity_rank_names);

    if (is_exodus_)
        mesh_data->add_mesh_database(
            file_name_, "exodusII", stk::io::READ_MESH);
    else
        mesh_data->add_mesh_database(file_name_, "cgns", stk::io::READ_MESH);

    mesh_data->create_input_mesh();
    auto metaData = Teuchos::rcp(mesh_data->meta_data_ptr());

    auto mesh = Teuchos::rcp(new panzer_stk::STK_Interface(metaData));
    mesh->initializeFromMetaData();
    mesh->instantiateBulkData(parallel_mach);
    mesh_data->set_bulk_data(Teuchos::get_shared_ptr(mesh->getBulkData()));

    // read in other transient fields, these will be useful later when
    // trying to read other fields for use in solve
    mesh_data->add_all_mesh_fields_as_input_fields();

    // store mesh data pointer for later use in initializing
    // bulk data
    mesh->getMetaData()->declare_attribute_with_delete(mesh_data);

    // build element blocks
    registerElementBlocks(*mesh, *mesh_data);
    registerSidesets(*mesh);
    registerNodesets(*mesh);

    mesh->addPeriodicBCs(this->periodicBCVec_);

    return mesh;
}

//---------------------------------------------------------------------------//
void StkReaderFactory::completeMeshConstruction(
    panzer_stk::STK_Interface& mesh, stk::ParallelMachine parallel_mach) const
{
    if (not mesh.isInitialized())
    {
        const bool buildRefinementSupport = levels_of_refinement_ > 0 ? true
                                                                      : false;
        mesh.initialize(parallel_mach, true, buildRefinementSupport);
    }

    // grab mesh data pointer to build the bulk data
    stk::mesh::MetaData& metaData = *mesh.getMetaData();
    const stk::mesh::BulkData& bulkData = *mesh.getBulkData();
    stk::io::StkMeshIoBroker* mesh_data = const_cast<stk::io::StkMeshIoBroker*>(
        metaData.get_attribute<stk::io::StkMeshIoBroker>());

    // remove the MeshData attribute
    TEUCHOS_ASSERT(metaData.remove_attribute(mesh_data));

    // build mesh bulk data
    mesh_data->populate_bulk_data();

    // Refine
    const bool delete_parent_elements = true;
    if (levels_of_refinement_ > 0)
        mesh.refineMesh(levels_of_refinement_, delete_parent_elements);

    // The following section of code is applicable if mesh scaling is
    // turned on from the input file.
    if (user_mesh_scaling_)
    {
        stk::mesh::Field<double>* coord_field = metaData.get_field<double>(
            stk::topology::NODE_RANK, "coordinates");
        const stk::mesh::Selector select_all_local
            = metaData.locally_owned_part() | metaData.globally_shared_part();
        const stk::mesh::BucketVector& my_node_buckets
            = bulkData.get_buckets(stk::topology::NODE_RANK, select_all_local);

        const int mesh_dim = mesh.getDimension();

        // Scale the mesh
        const double inv_msf = 1.0 / mesh_scale_factor_;
        for (size_t i = 0; i < my_node_buckets.size(); ++i)
        {
            const stk::mesh::Bucket& b = *(my_node_buckets[i]);
            double* coordinate_data = field_data(*coord_field, b);

            for (size_t j = 0; j < b.size(); ++j)
            {
                for (int k = 0; k < mesh_dim; ++k)
                {
                    coordinate_data[mesh_dim * j + k] *= inv_msf;
                }
            }
        }
    }

    // put in a negative index and (like python) the restart will be from the
    // back (-1 is the last time step)
    int restart_index = restart_index_;
    if (restart_index < 0)
    {
        const std::pair<int, double> lastTimeStep
            = mesh_data->get_input_ioss_region()->get_max_time();
        restart_index = 1 + restart_index + lastTimeStep.first;
    }

    // populate mesh fields with specific index
    mesh_data->read_defined_input_fields(restart_index);

    mesh.buildSubcells();
    mesh.buildLocalElementIDs();

    if (user_mesh_scaling_)
    {
        stk::mesh::Field<double>* coord_field = metaData.get_field<double>(
            stk::topology::NODE_RANK, "coordinates");
        std::vector<const stk::mesh::FieldBase*> fields;
        fields.push_back(coord_field);

        stk::mesh::communicate_field_data(bulkData, fields);
    }

    // process_input_request is a no-op if restart_index <= 0 ... thus there
    // would be no inital time
    if (restart_index > 0)
    {
        mesh.setInitialStateTime(
            mesh_data->get_input_ioss_region()->get_state_time(restart_index));
    }
    else
    {
        mesh.setInitialStateTime(0.0);
    }

    // clean up mesh data object
    delete mesh_data;

    // calls Stk_MeshFactory::rebalance
    this->rebalance(mesh);
}

//---------------------------------------------------------------------------//
//! From ParameterListAcceptor
void StkReaderFactory::setParameterList(
    const Teuchos::RCP<Teuchos::ParameterList>& param_list)
{
    TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(
        !param_list->isParameter("File Name"),
        Teuchos::Exceptions::InvalidParameterName,
        "Error, the parameter {name=\"File Name\","
        "type=\"string\""
        "\nis required in parameter (sub)list \""
            << param_list->name()
            << "\"."
               "\n\nThe parsed parameter parameter list is: \n"
            << param_list->currentParametersString());

    // Set default values here. Not all the params should be set so this
    // has to be done manually as opposed to using
    // validateParametersAndSetDefaults().
    if (!param_list->isParameter("Decomp Method"))
        param_list->set<std::string>("Decomp Method", "RIB");

    if (!param_list->isParameter("Restart Index"))
        param_list->set<int>("Restart Index", -1);

    if (!param_list->isParameter("File Type"))
        param_list->set("File Type", "Exodus");

    if (!param_list->isSublist("Periodic BCs"))
        param_list->sublist("Periodic BCs");

    Teuchos::ParameterList& p_bcs = param_list->sublist("Periodic BCs");
    if (!p_bcs.isParameter("Count"))
        p_bcs.set<int>("Count", 0);

    if (!param_list->isParameter("Levels of Uniform Refinement"))
        param_list->set<int>("Levels of Uniform Refinement", 0);

    param_list->validateParameters(*getValidParameters(), 0);

    setMyParamList(param_list);

    file_name_ = param_list->get<std::string>("File Name");

    decomp_method_ = param_list->get<std::string>("Decomp Method");

    restart_index_ = param_list->get<int>("Restart Index");

    const auto file_type = param_list->get<std::string>("File Type");
    is_exodus_ = (file_type == "Exodus");

    // get any mesh scale factor
    if (param_list->isParameter("Scale Factor"))
    {
        mesh_scale_factor_ = param_list->get<double>("Scale Factor");
        user_mesh_scaling_ = true;
    }

    // read in periodic boundary conditions
    parsePeriodicBCList(Teuchos::rcpFromRef(param_list->sublist("Periodic "
                                                                "BCs")),
                        this->periodicBCVec_,
                        this->useBBoxSearch_);

    levels_of_refinement_
        = param_list->get<int>("Levels of Uniform Refinement");
}

//---------------------------------------------------------------------------//
//! From ParameterListAcceptor
Teuchos::RCP<const Teuchos::ParameterList>
StkReaderFactory::getValidParameters() const
{
    static Teuchos::RCP<Teuchos::ParameterList> validParams;

    if (validParams == Teuchos::null)
    {
        validParams = Teuchos::rcp(new Teuchos::ParameterList);
        validParams->set<std::string>(
            "File Name",
            "<file name not set>",
            "Name of mesh file to be read",
            Teuchos::rcp(new Teuchos::FileNameValidator));

        validParams->set<std::string>("Decomp Method",
                                      "RIB",
                                      "Parallel mesh decomposition method",
                                      Teuchos::rcp(new Teuchos::StringValidator(
                                          Ioss::valid_decomp_methods())));

        validParams->set<int>(
            "Restart Index",
            -1,
            "Index of solution to read in",
            Teuchos::rcp(new Teuchos::AnyNumberParameterEntryValidator(
                Teuchos::AnyNumberParameterEntryValidator::PREFER_INT,
                Teuchos::AnyNumberParameterEntryValidator::AcceptedTypes(
                    true))));

        Teuchos::setStringToIntegralParameter<int>(
            "File Type",
            "Exodus",
            "Choose input file type - either \"Exodus\" or \"CGNS\"",
            Teuchos::tuple<std::string>("Exodus", "CGNS"),
            validParams.get());

        validParams->set<double>(
            "Scale Factor",
            1.0,
            "Scale factor to apply to mesh after read",
            Teuchos::rcp(new Teuchos::AnyNumberParameterEntryValidator(
                Teuchos::AnyNumberParameterEntryValidator::PREFER_DOUBLE,
                Teuchos::AnyNumberParameterEntryValidator::AcceptedTypes(
                    true))));

        Teuchos::ParameterList& bcs = validParams->sublist("Periodic BCs");
        bcs.set<int>("Count", 0); // no default periodic boundary conditions

        validParams->set("Levels of Uniform Refinement",
                         0,
                         "Number of levels of inline uniform mesh refinement");
    }

    return validParams.getConst();
}

//---------------------------------------------------------------------------//
void StkReaderFactory::registerElementBlocks(
    panzer_stk::STK_Interface& mesh, stk::io::StkMeshIoBroker& mesh_data) const
{
    auto femMetaData = mesh.getMetaData();

    // here we use the Ioss interface because they don't add
    // "bonus" element blocks and its easier to determine
    // "real" element blocks versus STK-only blocks
    const Ioss::ElementBlockContainer& elem_blocks
        = mesh_data.get_input_ioss_region()->get_element_blocks();
    for (const auto& entity : elem_blocks)
    {
        // Ioss::GroupingEntity* entity = *itr;
        const std::string& name = entity->name();

        const stk::mesh::Part* part = femMetaData->get_part(name);
        const shards::CellTopology cellTopo
            = stk::mesh::get_cell_topology(femMetaData->get_topology(*part));
        const CellTopologyData* ct = cellTopo.getCellTopologyData();

        TEUCHOS_ASSERT(ct != 0);
        mesh.addElementBlock(part->name(), ct);
    }
}

//---------------------------------------------------------------------------//
void StkReaderFactory::registerSidesets(panzer_stk::STK_Interface& mesh) const
{
    const Teuchos::RCP<stk::mesh::MetaData> metaData = mesh.getMetaData();
    const stk::mesh::PartVector& parts = metaData->get_parts();

    for (const auto& part : parts)
    {
        const stk::mesh::PartVector& subsets = part->subsets();
        const shards::CellTopology cellTopo
            = stk::mesh::get_cell_topology(metaData->get_topology(*part));
        const CellTopologyData* ct = cellTopo.getCellTopologyData();

        // if a side part ==> this is a sideset: now storage is recursive
        // on part contains all sub parts with consistent topology
        if (part->primary_entity_rank() == mesh.getSideRank() && ct == 0
            && subsets.size() > 0)
        {
            TEUCHOS_TEST_FOR_EXCEPTION(
                subsets.size() != 1,
                std::runtime_error,
                "StkReaderFactory::registerSidesets error - part \""
                    << part->name() << "\" has more than one subset");

            // grab cell topology and name of subset part
            const stk::mesh::Part* ss_part = subsets[0];
            const shards::CellTopology ss_cellTopo
                = stk::mesh::get_cell_topology(
                    metaData->get_topology(*ss_part));
            const CellTopologyData* ss_ct = ss_cellTopo.getCellTopologyData();

            // only add subset parts that have no topology
            if (ss_ct != 0)
                mesh.addSideset(part->name(), ss_ct);
        }
    }
}

//---------------------------------------------------------------------------//
void StkReaderFactory::registerNodesets(panzer_stk::STK_Interface& mesh) const
{
    const Teuchos::RCP<stk::mesh::MetaData> metaData = mesh.getMetaData();
    const stk::mesh::PartVector& parts = metaData->get_parts();

    stk::mesh::PartVector::const_iterator partItr;
    for (partItr = parts.begin(); partItr != parts.end(); ++partItr)
    {
        const stk::mesh::Part* part = *partItr;
        const shards::CellTopology cellTopo
            = stk::mesh::get_cell_topology(metaData->get_topology(*part));
        const CellTopologyData* ct = cellTopo.getCellTopologyData();

        // if a side part ==> this is a sideset: now storage is recursive
        // on part contains all sub parts with consistent topology
        if (part->primary_entity_rank() == mesh.getNodeRank() && ct == 0)
        {
            // only add subset parts that have no topology
            if (part->name() != panzer_stk::STK_Interface::nodesString)
                mesh.addNodeset(part->name());
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace Mesh
} // end namespace VertexCFD
