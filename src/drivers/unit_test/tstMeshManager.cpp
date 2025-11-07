#include <VertexCFD_DriverUnitTestConfig.hpp>

#include <drivers/VertexCFD_MeshManager.hpp>
#include <parameters/VertexCFD_ParameterDatabase.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
TEST(MeshManager, file_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Make an empty parameter database and build mesh parmaeters.
    const Parameter::ParameterDatabase parameter_db(comm);
    auto mesh_params = parameter_db.meshParameters();
    mesh_params->set("Mesh Input Type", "File");
    auto& file_params = mesh_params->sublist("File");
    const std::string mesh_location = VERTEXCFD_DRIVER_TEST_MESH_DIR;
    const std::string mesh_file = "test_mesh_manager.exo";
    const std::string filepath = mesh_location + mesh_file;
    file_params.set("File Name", filepath);
    file_params.set("Decomp Method", "RCB");

    // Create the mesh.
    MeshManager mesh_manager(parameter_db, comm);
    mesh_manager.completeMeshConstruction();

    // Check the mesh.
    EXPECT_EQ(2, mesh_manager.spaceDimension());
    auto mesh = mesh_manager.mesh();
    EXPECT_EQ(2, mesh->getNumElementBlocks());

    // Check the connectivity by checking the element block.
    auto conn = mesh_manager.connectivityManager();
    EXPECT_EQ(2, conn->numElementBlocks());
}

//---------------------------------------------------------------------------//
void testInlineMesh(const std::string element_type)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Mesh dimension based on 'element_type' entry. 'split' is
    // multiplicator used to scale the number of elements when
    // using triangle or tetrahedral elements
    int mesh_dimension = 0;
    int split = 1;
    if (element_type == "Quad4" || element_type == "Tri3")
    {
        mesh_dimension = 2;
        if (element_type == "Tri3")
            split = 2;
    }
    else if (element_type == "Tet4" || element_type == "Hex8")
    {
        mesh_dimension = 3;
        if (element_type == "Tet4")
            split = 12;
    }

    // Make an empty parameter database and build mesh parameters.
    const Parameter::ParameterDatabase parameter_db(comm);
    auto mesh_params = parameter_db.meshParameters();
    mesh_params->set("Mesh Input Type", "Inline");
    auto& inline_params = mesh_params->sublist("Inline");
    inline_params.set("Element Type", element_type);
    auto& mesh_details = inline_params.sublist("Mesh");
    const int nelem_x = 3;
    const int nelem_y = nelem_x * mesh_dimension;
    const int nelem_z = nelem_y * mesh_dimension;
    mesh_details.set("X0", 0.0);
    mesh_details.set("Xf", 1.0);
    mesh_details.set("X Elements", nelem_x);
    if (mesh_dimension > 1)
    {
        mesh_details.set("Y0", 0.0);
        mesh_details.set("Yf", 1.0);
        mesh_details.set("Y Elements", nelem_y);
    }
    if (mesh_dimension > 2)
    {
        mesh_details.set("Z0", 0.0);
        mesh_details.set("Zf", 1.0);
        mesh_details.set("Z Elements", nelem_z);
    }

    // Create the mesh.
    MeshManager mesh_manager(parameter_db, comm);
    mesh_manager.completeMeshConstruction();

    // Check the mesh.
    EXPECT_EQ(mesh_dimension, mesh_manager.spaceDimension());
    const auto mesh = mesh_manager.mesh();
    EXPECT_EQ(1, mesh->getNumElementBlocks());
    int mesh_num_elem = split * nelem_x;
    if (mesh_dimension > 1)
        mesh_num_elem *= nelem_y;
    if (mesh_dimension > 2)
        mesh_num_elem *= nelem_z;
    EXPECT_EQ(mesh_num_elem, mesh->getEntityCounts(stk::topology::ELEM_RANK));

    // Check the connectivity by checking the element block.
    auto conn = mesh_manager.connectivityManager();
    EXPECT_EQ(1, conn->numElementBlocks());
}
//---------------------------------------------------------------------------//
TEST(MeshManager, tri3_test)
{
    testInlineMesh("Tri3");
}
//---------------------------------------------------------------------------//
TEST(MeshManager, quad4_test)
{
    testInlineMesh("Quad4");
}
//---------------------------------------------------------------------------//
TEST(MeshManager, tet4_test)
{
    testInlineMesh("Tet4");
}
//---------------------------------------------------------------------------//
TEST(MeshManager, hex8_test)
{
    testInlineMesh("Hex8");
}
//---------------------------------------------------------------------------//
TEST(MeshManager, bad_elem_type_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Make an empty parameter database and build mesh parmaeters.
    const Parameter::ParameterDatabase parameter_db(comm);
    auto mesh_params = parameter_db.meshParameters();
    mesh_params->set("Mesh Input Type", "Inline");
    auto& inline_params = mesh_params->sublist("Inline");
    inline_params.set("Element Type", "BadElem");

    // Create the mesh.
    const std::string msg
        = "Invalid inline element type. Valid options are 'Tri3', "
          "'Quad4', 'Tet4' and 'Hex8'";
    EXPECT_THROW(
        try {
            const MeshManager mesh_manager(parameter_db, comm);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::runtime_error);
}

//---------------------------------------------------------------------------//
TEST(MeshManager, bad_mesh_type_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Make an empty parameter database and build mesh parmaeters.
    const Parameter::ParameterDatabase parameter_db(comm);
    auto mesh_params = parameter_db.meshParameters();
    mesh_params->set("Mesh Input Type", "Throw");

    // Create the mesh.
    const std::string msg
        = "Invalid mesh input type. Valid options are 'File' and 'Inline'";
    EXPECT_THROW(
        try {
            const MeshManager mesh_manager(parameter_db, comm);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::runtime_error);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
