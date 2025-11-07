#ifndef VERTEXCFD_PARAMETERDATABASE_HPP
#define VERTEXCFD_PARAMETERDATABASE_HPP

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <string>

namespace VertexCFD
{
namespace Parameter
{
//---------------------------------------------------------------------------//
class ParameterDatabase
{
  public:
    // Default constructor.
    ParameterDatabase(const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm);

    // Parameter list constructor.
    ParameterDatabase(const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
                      const Teuchos::RCP<Teuchos::ParameterList>& parameters);

    // XML file constructor.
    ParameterDatabase(const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
                      const std::string& xml_file);

    // Main argument constructor.
    ParameterDatabase(const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
                      int argc,
                      char* argv[]);

    // Communicator.
    Teuchos::RCP<const Teuchos::MpiComm<int>> comm() const;

    // Main list accessor.
    Teuchos::RCP<Teuchos::ParameterList> allParameters() const;

    // Sublist accessors.
    Teuchos::RCP<Teuchos::ParameterList> meshParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> assemblyParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> scalarParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> generalScalarParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> boundaryConditionParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> initialConditionParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> closureModelParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> responseOutputParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> userParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> outputParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> readRestartParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> writeRestartParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> writeMatrixParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> profilingParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> transientSolverParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> linearSolverParameters() const;

    // Deprecated sublist accessors.
    Teuchos::RCP<Teuchos::ParameterList> physicsParameters() const;
    Teuchos::RCP<Teuchos::ParameterList> blockMappingParameters() const;

    // New format boolean. This will be removed once the transition to the new
    // format is complete.
    bool useNewInputFormat() const;

  private:
    // Read an xml file with parameters and extract sublists.
    void readParameterXmlFile(const std::string& xml_file);

    // Read an yaml file with parameters and extract sublists.
    void readParameterYamlFile(const std::string& xml_file);

    // Get the sublists from the input parameters.
    void extractSublists();

    // Get the sublists from the input parameters - old variant. (Deprecated)
    void extractSublistsOld();

    // Get the sublists from the input parameters - new variant.
    void extractSublistsNew();

    // Get a required sublist from the main input list.
    Teuchos::RCP<Teuchos::ParameterList>
    requiredSublist(const std::string& name);

    // Get an optional sublist from the main input list.
    Teuchos::RCP<Teuchos::ParameterList>
    optionalSublist(const std::string& name);

  private:
    Teuchos::RCP<const Teuchos::MpiComm<int>> _comm;
    Teuchos::RCP<Teuchos::ParameterList> _input_params;
    Teuchos::RCP<Teuchos::ParameterList> _mesh_params;
    Teuchos::RCP<Teuchos::ParameterList> _assembly_params;
    Teuchos::RCP<Teuchos::ParameterList> _scalar_params;
    Teuchos::RCP<Teuchos::ParameterList> _general_scalar_params;
    Teuchos::RCP<Teuchos::ParameterList> _bc_params;
    Teuchos::RCP<Teuchos::ParameterList> _ic_params;
    Teuchos::RCP<Teuchos::ParameterList> _closure_params;
    Teuchos::RCP<Teuchos::ParameterList> _response_output_params;
    Teuchos::RCP<Teuchos::ParameterList> _user_params;
    Teuchos::RCP<Teuchos::ParameterList> _output_params;
    Teuchos::RCP<Teuchos::ParameterList> _read_restart_params;
    Teuchos::RCP<Teuchos::ParameterList> _write_restart_params;
    Teuchos::RCP<Teuchos::ParameterList> _write_matrix_params;
    Teuchos::RCP<Teuchos::ParameterList> _profiling_params;
    Teuchos::RCP<Teuchos::ParameterList> _transient_solver_params;
    Teuchos::RCP<Teuchos::ParameterList> _linear_solver_params;
    bool _use_new_input = false;

    // Deprecated lists.
    Teuchos::RCP<Teuchos::ParameterList> _physics_params;
    Teuchos::RCP<Teuchos::ParameterList> _block_mapping_params;
};

//---------------------------------------------------------------------------//

} // namespace Parameter
} // namespace VertexCFD

#endif // end VERTEXCFD_PARAMETERDATABASE_HPP
