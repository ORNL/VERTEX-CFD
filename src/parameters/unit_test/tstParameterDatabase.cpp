#include "VertexCFD_ParameterUnitTestConfig.hpp"

#include "parameters/VertexCFD_ParameterDatabase.hpp"

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
void testDefaultDatabase()
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Create database.
    const VertexCFD::Parameter::ParameterDatabase parameter_db(comm);

    // Check that all parameter lists got populated.
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.meshParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.physicsParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.blockMappingParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.scalarParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.generalScalarParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.boundaryConditionParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.sortedBoundaryConditions()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.initialConditionParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.closureModelParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.responseOutputParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.userParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.outputParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.readRestartParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.writeRestartParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.writeMatrixParameters()));
    EXPECT_TRUE(Teuchos::is_null(parameter_db.profilingParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.transientSolverParameters()));
    EXPECT_TRUE(Teuchos::nonnull(parameter_db.linearSolverParameters()));

    // Check that we added the communicator to the user data.
    const auto param_comm
        = parameter_db.userParameters()
              ->get<Teuchos::RCP<const Teuchos::Comm<int>>>("Comm");
    EXPECT_EQ(comm->getRank(), param_comm->getRank());
    EXPECT_EQ(comm->getSize(), param_comm->getSize());
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, default_test)
{
    testDefaultDatabase();
}

//---------------------------------------------------------------------------//
void testInputParser(const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
                     const VertexCFD::Parameter::ParameterDatabase& parameter_db,
                     const bool new_input_format = false)
{
    // Check that all parameter lists got populated.
    EXPECT_DOUBLE_EQ(1.0, parameter_db.meshParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.4,
                     parameter_db.physicsParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.3, parameter_db.blockMappingParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.6,
                     parameter_db.sortedBoundaryConditions()
                         ->sublist("child0")
                         .sublist("Data")
                         .get<double>("value"));
    EXPECT_DOUBLE_EQ(3.2,
                     parameter_db.sortedBoundaryConditions()
                         ->sublist("child1")
                         .sublist("Data")
                         .get<double>("value"));
    EXPECT_DOUBLE_EQ(1.6,
                     parameter_db.sortedBoundaryConditions()
                         ->sublist("child2")
                         .sublist("Data")
                         .get<double>("value"));
    EXPECT_DOUBLE_EQ(3.2,
                     parameter_db.sortedBoundaryConditions()
                         ->sublist("child3")
                         .sublist("Data")
                         .get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.7, parameter_db.initialConditionParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.8, parameter_db.closureModelParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.4, parameter_db.responseOutputParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.5, parameter_db.userParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.1,
                     parameter_db.outputParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.3, parameter_db.readRestartParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.2, parameter_db.writeRestartParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.1, parameter_db.writeMatrixParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(1.2,
                     parameter_db.profilingParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.0, parameter_db.transientSolverParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        1.9, parameter_db.linearSolverParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(2.5,
                     parameter_db.scalarParameters()->get<double>("value"));
    EXPECT_DOUBLE_EQ(
        2.6, parameter_db.generalScalarParameters()->get<double>("value"));
    if (new_input_format)
    {
        EXPECT_DOUBLE_EQ(
            2.7, parameter_db.assemblyParameters()->get<double>("value"));
    }

    // Check the parameter communicator.
    EXPECT_EQ(comm->getRank(), parameter_db.comm()->getRank());
    EXPECT_EQ(comm->getSize(), parameter_db.comm()->getSize());

    // Check that we added the communicator to the user data.
    const auto param_comm
        = parameter_db.userParameters()
              ->get<Teuchos::RCP<const Teuchos::Comm<int>>>("Comm");
    EXPECT_EQ(comm->getRank(), param_comm->getRank());
    EXPECT_EQ(comm->getSize(), param_comm->getSize());
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, argv_test_wrong_format)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 2;
    const std::string option = "--i=";
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.csv";
    std::string argv_str = option + location + file;
    char* argv[2];
    argv[1] = &argv_str[0];

    const std::string exp_msg
        = "\n\nERROR: Vertex-CFD only supports input files of XML and YMAL "
          "formats.\n\n";

    EXPECT_THROW(
        try {
            const VertexCFD::Parameter::ParameterDatabase parameter_db(
                comm, argc, argv);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
}
//---------------------------------------------------------------------------//
TEST(ParameterDatabase, argv_test_xml)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 3;
    const std::string option = "--i=";
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.xml";
    std::string argv_str = option + location + file;
    std::string argv_str_two = "--no-conversion-to-yaml";
    char* argv[3];
    argv[1] = &argv_str[0];
    argv[2] = &argv_str_two[0];
    const VertexCFD::Parameter::ParameterDatabase parameter_db(
        comm, argc, argv);

    // Test output
    testInputParser(comm, parameter_db);

    // Check that the generated file is not there
    argv_str = location + "input_parser_test_generated.yaml";
    try
    {
        ASSERT_TRUE(fopen(&argv_str[0], "r") == NULL);
    }
    catch (const std::runtime_error& e)
    {
        std::remove(&argv_str[0]);
        std::cout << "Error: " << e.what() << std::endl;
    }
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, argv_test_yaml)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 3;
    const std::string option = "--i=";
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.yaml";
    std::string argv_str = option + location + file;
    std::string argv_str_two = "--no-conversion-to-xml";
    char* argv[3];
    argv[1] = &argv_str[0];
    argv[2] = &argv_str_two[0];
    const VertexCFD::Parameter::ParameterDatabase parameter_db(
        comm, argc, argv);

    // Test output
    testInputParser(comm, parameter_db);

    // Check that the generated file is not there
    argv_str = location + "input_parser_test_generated.xml";
    try
    {
        ASSERT_TRUE(fopen(&argv_str[0], "r") == NULL);
    }
    catch (const std::runtime_error& e)
    {
        std::remove(&argv_str[0]);
        std::cout << "Error: " << e.what() << std::endl;
    }
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, argv_test_generated_xml)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 3;
    const std::string option = "--i=";
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.xml";
    std::string argv_str = option + location + file;
    std::string argv_str_two = "--conversion-to-yaml";
    char* argv[3];
    argv[1] = &argv_str[0];
    argv[2] = &argv_str_two[0];
    const VertexCFD::Parameter::ParameterDatabase parameter_db(
        comm, argc, argv);

    // Test output
    testInputParser(comm, parameter_db);

    // Compare generated file against gold file and delete generated file if
    // successfull
    try
    {
        argv_str = location + "input_parser_test_generated_gold.yaml";
        std::ifstream file_ifstream_gold(&argv_str[0]);
        const std::string file_str_gold(
            (std::istreambuf_iterator<char>(file_ifstream_gold)),
            std::istreambuf_iterator<char>());

        argv_str = location + "input_parser_test_generated.yaml";
        std::ifstream file_ifstream(&argv_str[0]);
        const std::string file_str(
            (std::istreambuf_iterator<char>(file_ifstream)),
            std::istreambuf_iterator<char>());
        ASSERT_EQ(file_str_gold, file_str);
        std::remove(&argv_str[0]);
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, argv_test_generated_yaml)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 3;
    const std::string option = "--i=";
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.yaml";
    std::string argv_str = option + location + file;
    std::string argv_str_two = "--conversion-to-xml";
    char* argv[3];
    argv[1] = &argv_str[0];
    argv[2] = &argv_str_two[0];
    const VertexCFD::Parameter::ParameterDatabase parameter_db(
        comm, argc, argv);

    // Test output
    testInputParser(comm, parameter_db);

    // Compare generated file against gold file and delete generated file if
    // successfull
    try
    {
        argv_str = location + "input_parser_test_generated_gold.xml";
        std::ifstream file_ifstream_gold(&argv_str[0]);
        const std::string file_str_gold(
            (std::istreambuf_iterator<char>(file_ifstream_gold)),
            std::istreambuf_iterator<char>());

        argv_str = location + "input_parser_test_generated.xml";
        std::ifstream file_ifstream(&argv_str[0]);
        const std::string file_str(
            (std::istreambuf_iterator<char>(file_ifstream)),
            std::istreambuf_iterator<char>());
        ASSERT_EQ(file_str_gold, file_str);
        std::remove(&argv_str[0]);
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, file_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.xml";
    const std::string filename = location + file;
    const VertexCFD::Parameter::ParameterDatabase parameter_db(comm, filename);

    // Test
    testInputParser(comm, parameter_db);
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, list_test)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test.xml";
    const std::string filename = location + file;
    const VertexCFD::Parameter::ParameterDatabase file_db(comm, filename);

    // Create a new parameter database from the input list.
    const VertexCFD::Parameter::ParameterDatabase parameter_db(
        comm, file_db.allParameters());

    // Test
    testInputParser(comm, parameter_db);
}

//---------------------------------------------------------------------------//
TEST(ParameterDatabase, list_test_new)
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const std::string location = VERTEXCFD_PARAMETER_TEST_DATA_DIR;
    const std::string file = "input_parser_test_new.xml";
    const std::string filename = location + file;
    const VertexCFD::Parameter::ParameterDatabase file_db(comm, filename);

    // Create a new parameter database from the input list.
    auto parameter_list = file_db.allParameters();
    parameter_list->set<bool>("Use New Input Format", true);
    const VertexCFD::Parameter::ParameterDatabase parameter_db(comm,
                                                               parameter_list);

    // Test
    testInputParser(comm, parameter_db, true);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
