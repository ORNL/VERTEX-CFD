// Due to a conflict with FAD types and Kokkos views, this file needs to be
// included before other VertexCFD includes
#include "utils/VertexCFD_Utils_KokkosFadFixup.hpp"

#include "VertexCFD_ExternalFieldsManager.hpp"
#include "VertexCFD_InitialConditionManager.hpp"
#include "VertexCFD_MeshManager.hpp"
#include "VertexCFD_PhysicsManager.hpp"

#include "mesh/VertexCFD_Mesh_ExodusWriter.hpp"
#include "mesh/VertexCFD_Mesh_Restart.hpp"
#include "observers/VertexCFD_Compute_ErrorNorms.hpp"
#include "observers/VertexCFD_Compute_Volume.hpp"
#include "observers/VertexCFD_NOXObserver_IterationOutput.hpp"
#include "observers/VertexCFD_TempusObserver_ErrorNormOutput.hpp"
#include "observers/VertexCFD_TempusObserver_IterationOutput.hpp"
#include "observers/VertexCFD_TempusObserver_ResponseOutput.hpp"
#include "observers/VertexCFD_TempusObserver_WriteMatrix.hpp"
#include "observers/VertexCFD_TempusObserver_WriteRestart.hpp"
#include "observers/VertexCFD_TempusObserver_WriteToExodus.hpp"
#include "observers/VertexCFD_TempusTimeStepControl_GlobalCFL.hpp"
#include "observers/VertexCFD_TempusTimeStepControl_GlobalTimeStep.hpp"
#include "observers/VertexCFD_TempusTimeStepControl_Strategy.hpp"
#include "parameters/VertexCFD_ParameterDatabase.hpp"
#include "responses/VertexCFD_ResponseManager.hpp"
#include "responses/VertexCFD_Response_Utils.hpp"

#include <Trilinos_version.h>

#include <Tempus_IntegratorBasic.hpp>
#include <Tempus_IntegratorObserverComposite.hpp>

#include <NOX.H>
#include <NOX_Observer_Vector.hpp>
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
#include <NOX_Observer_ReusePreconditionerFactory.hpp>
#endif

#include <Panzer_InitialCondition_Builder.hpp>
#include <Panzer_NodeType.hpp>
#include <Panzer_PauseToAttach.hpp>
#include <Panzer_ResponseLibrary.hpp>
#include <Panzer_String_Utilities.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

//---------------------------------------------------------------------------//
// Parse input file and initialize mesh manager
void parse_input_file_and_mesh(
    int argc,
    char* argv[],
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    Teuchos::RCP<VertexCFD::Parameter::ParameterDatabase>& parameter_db,
    Teuchos::RCP<VertexCFD::MeshManager>& mesh_manager)
{
    // Parse input file specified in 'argc' and store its content in
    // 'parameter_db'
    parameter_db = Teuchos::rcp(
        new VertexCFD::Parameter::ParameterDatabase(comm, argc, argv));

    // Initialize mesh manager
    mesh_manager
        = Teuchos::rcp(new VertexCFD::MeshManager(*parameter_db, comm));
}

//---------------------------------------------------------------------------//
// Run VertexCFD after parsing the input file and initializinig the mesh
// manager object in function 'parse_input_file_and_mesh'.
template<int NumSpaceDim>
void run_vertexcfd(
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    const Teuchos::RCP<VertexCFD::Parameter::ParameterDatabase>& parameter_db,
    const Teuchos::RCP<VertexCFD::MeshManager>& mesh_manager)
{
    // Setup an output stream that will only write to rank 0.
    Teuchos::FancyOStream ostream(Teuchos::rcpFromRef(std::cout));
    ostream.setShowProcRank(false);
    ostream.setOutputToRootOnly(0);

    // Get sublists.
    auto solver_params = parameter_db->transientSolverParameters();
    auto closure_params = parameter_db->closureModelParameters();
    auto response_output_params = parameter_db->responseOutputParameters();
    auto user_params = parameter_db->userParameters();
    auto output_params = parameter_db->outputParameters();
    auto write_restart_params = parameter_db->writeRestartParameters();
    auto write_matrix_params = parameter_db->writeMatrixParameters();
    auto profiling_params = parameter_db->profilingParameters();

    // Setup timers.
    bool use_timers = Teuchos::nonnull(profiling_params);
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    if (use_timers)
    {
        stacked_timer = Teuchos::rcp(new Teuchos::StackedTimer("VertexCFD"));
        Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
        stacked_timer->start("Physics");
    }

    // Used for template argument deduction in constructors below.
    constexpr auto num_space_dim = std::integral_constant<int, NumSpaceDim>{};

    // Create external field evaluator if needed
    if (user_params->isType<std::string>("External Field Parameter File"))
    {
        const std::string ef_filename
            = user_params->get<std::string>("External Field Parameter File");
        auto external_fields_manager = Teuchos::rcp(
            new VertexCFD::ExternalFieldsManager<panzer::Traits>(
                num_space_dim, comm, ef_filename));
        user_params->set("External Fields Manager", external_fields_manager);
    }

    // Get mesh object from mesh manager.
    auto mesh = mesh_manager->mesh();

    // Create initial condition manager.
    auto ic_manager = Teuchos::rcp(
        new VertexCFD::InitialConditionManager(parameter_db, mesh_manager));
    auto t_init = ic_manager->initialTime();

    // Create physics.
    auto physics_manager = Teuchos::rcp(new VertexCFD::PhysicsManager(
        num_space_dim, parameter_db, mesh_manager, t_init));
    auto physics = physics_manager->modelEvaluator();
    auto integration_order = physics_manager->integrationOrder();

    // Setup volume/surface responses.
    std::vector<int> response_output_freq;
    auto responses = Teuchos::rcp(
        new VertexCFD::Response::ResponseManager(physics_manager));
    if (Teuchos::nonnull(response_output_params))
    {
        if (response_output_params->numParams() > 0)
        {
            // Allow setting output frequency, defaulting to once at the end.
            const auto default_output_freq = response_output_params->get<int>(
                "Output Frequency", std::numeric_limits<int>::max());

            for (auto param_itr = response_output_params->begin();
                 param_itr != response_output_params->end();
                 ++param_itr)
            {
                const auto& name = param_itr->first;

                // Skip over any regular Parameters.
                if (!response_output_params->isSublist(name))
                    continue;

                auto& plist = response_output_params->sublist(name);

                const auto field_names_list
                    = plist.get<std::string>("Field Name");
                std::vector<std::string> field_names;
                panzer::StringTokenizer(
                    field_names, field_names_list, ",", true);
                const int num_fields = field_names.size();

                // Allow overriding output frequency for this response.
                const auto output_freq
                    = plist.get<int>("Output Frequency", default_output_freq);

                // Get element blocks or sidesets for this response.
                const auto workset_descriptors
                    = VertexCFD::Response::buildWorksetDescriptors(plist);

                // Add the response and save response output frequency
                if (plist.isSublist("Probe "
                                    "Coordinates"))
                {
                    const auto probe_list = plist.sublist("Probe Coordinates");
                    for (int j = 0; j < num_fields; ++j)
                    {
                        for (int i = 0; i < probe_list.numParams(); ++i)
                        {
                            const std::string si = std::to_string(i + 1);
                            const std::string pb_nm = "Probe " + si;
                            const std::string name_ji = name + " " + si + " - "
                                                        + field_names[j];
                            const auto point_i
                                = probe_list.get<Teuchos::Array<double>>(pb_nm);

                            responses->addProbeResponse(name_ji,
                                                        field_names[j],
                                                        point_i,
                                                        workset_descriptors);

                            response_output_freq.emplace_back(output_freq);
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < num_fields; ++j)
                    {
                        const std::string name_j = name + " - "
                                                   + field_names[j];
                        if (std::string::npos != name.find("Max"))
                        {
                            responses->addMaxValueResponse(
                                name_j, field_names[j], workset_descriptors);
                        }
                        else if (std::string::npos != name.find("Min"))
                        {
                            responses->addMinValueResponse(
                                name_j, field_names[j], workset_descriptors);
                        }
                        else
                        {
                            responses->addFunctionalResponse(
                                name_j, field_names[j], workset_descriptors);
                        }
                        response_output_freq.emplace_back(output_freq);
                    }
                }
            }
        }
    }

    // Setup time step control. This adds a response, so must be built before
    // calling setupModel.
    Teuchos::RCP<VertexCFD::TempusTimeStepControl::Strategy<double>> dt_strategy;
    if (user_params->isParameter("CFL"))
    {
        dt_strategy = Teuchos::rcp(
            new VertexCFD::TempusTimeStepControl::GlobalCFL<double>(
                *user_params, physics_manager));
    }
    else
    {
        dt_strategy = Teuchos::rcp(
            new VertexCFD::TempusTimeStepControl::GlobalTimeStep<double>(
                *user_params, physics_manager));
    }

    // Setup the model after the physics responses have been added.
    physics_manager->setupModel();
    auto workset_container = physics_manager->worksetContainer();
    auto dof_manager = physics_manager->dofManager();
    auto linear_object_factory = physics_manager->linearObjectFactory();
    auto cm_factory = physics_manager->closureModelFactory();
    auto physics_blocks = physics_manager->physicsBlocks();

    // Create io response library.
    Teuchos::RCP<VertexCFD::Mesh::ExodusWriter> exodus_writer;
    if (Teuchos::nonnull(output_params))
    {
        auto io_response_library
            = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(
                workset_container, dof_manager, linear_object_factory));

        // Make sure these io sublists exist.
        output_params->sublist("Cell Average Quantities");
        output_params->sublist("Cell Average Vectors");
        output_params->sublist("Cell Quantities");
        output_params->sublist("Nodal Quantities");

        // Setup solution output. This must occur before we build evaluators in
        // the response library so this observer may register its evaluators.
        exodus_writer = Teuchos::rcp(
            new VertexCFD::Mesh::ExodusWriter(mesh,
                                              dof_manager,
                                              linear_object_factory,
                                              io_response_library,
                                              *output_params));

        // Create io response evaluators.
        panzer_stk::IOClosureModelFactory_TemplateBuilder<panzer::Traits>
            io_cm_builder(*cm_factory, mesh, *output_params);
        panzer::ClosureModelFactory_TemplateManager<panzer::Traits> io_cm_factory;
        io_cm_factory.buildObjects(io_cm_builder);
        io_response_library->buildResponseEvaluators(
            physics_blocks, io_cm_factory, *closure_params, *user_params);
    }

    // Set initial conditions.
    Teuchos::RCP<Thyra::VectorBase<double>> solution;
    Teuchos::RCP<Thyra::VectorBase<double>> solution_dot;
    ic_manager->applyInitialConditions(
        num_space_dim, *physics_manager, solution, solution_dot);

    // Make vector of observers
    auto nox_observer_vector = Teuchos::rcp(new NOX::ObserverVector());

    // Setup iteration output observers.
    auto nox_iteration_observer
        = Teuchos::rcp(new VertexCFD::NOXObserver::IterationOutput());
    nox_observer_vector->pushBack(nox_iteration_observer);

    // If available/requested, make an observer to reuse preconditioner
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
    auto& nox_params = solver_params->sublist("Default Stepper")
                           .sublist("Default Solver")
                           .sublist("NOX");
    if (nox_params.isSublist("Reuse Preconditioner"))
    {
        auto nox_prec_reuse_observer = NOX::createReusePreconditionerObserver(
            nox_params.sublist("Reuse Preconditioner"));
        nox_observer_vector->pushBack(nox_prec_reuse_observer);
    }
#endif

    // Register observer vector with NOX parameters
    solver_params->sublist("Default Stepper")
        .sublist("Default Solver")
        .sublist("NOX")
        .sublist("Solver Options")
        .set<Teuchos::RCP<NOX::Observer>>("User Defined Pre/Post Operator",
                                          nox_observer_vector);

    // Setup time integrator -- toggle interface on Trilinos version
#if TRILINOS_MAJOR_MINOR_VERSION >= 130100
    // Remove Tempus entries that are deprecated in Trilinos 13.2
    auto integrator_params
        = Teuchos::sublist(solver_params, "Default Integrator");
    auto tsc_params = Teuchos::sublist(integrator_params, "Time Step Control");
    tsc_params->remove("Minimum Order", false);
    tsc_params->remove("Maximum Order", false);
    tsc_params->remove("Initial Order", false);
    tsc_params->remove("Integrator Step Type", false);
    auto integrator
        = Tempus::createIntegratorBasic<double>(solver_params, physics);
#else
    auto integrator = Tempus::integratorBasic<double>(solver_params, physics);
#endif

    // Build a composite observer containing all of our tempus observers.
    auto integrator_observer
        = Teuchos::rcp(new Tempus::IntegratorObserverComposite<double>());

    // Setup exodus output observer.
    if (Teuchos::nonnull(output_params))
    {
        auto exodus_observer = Teuchos::rcp(
            new VertexCFD::TempusObserver::WriteToExodus<double>(
                exodus_writer, *output_params));
        integrator_observer->addObserver(exodus_observer);
    }

    // Error norm response library
    auto error_norm_response_library
        = Teuchos::rcp(new panzer::ResponseLibrary<panzer::Traits>(
            workset_container, dof_manager, linear_object_factory));

    const bool error_norm_flag = user_params->isSublist("Compute Error Norms");
    if (error_norm_flag)
    {
        const auto bcs = physics_manager->boundaryConditions();
        const auto bc_factory = physics_manager->boundaryConditionFactory();
        const auto eq_set_factory = physics_manager->equationSetFactory();
        const auto error_norm_list
            = user_params->sublist("Compute Error Norms");

        // Compute Volume integration when Compute Error Norm
        auto volume = Teuchos::rcp(new VertexCFD::ComputeVolume::Volume<double>(
            mesh,
            linear_object_factory,
            error_norm_response_library,
            physics_blocks,
            *cm_factory,
            *closure_params,
            *user_params,
            workset_container,
            bcs,
            bc_factory,
            eq_set_factory,
            integration_order));
        volume->ComputeVol();
        const double volume_value = volume->volume();

        // L1/L2 Error norm instance
        auto comp_error_norms = Teuchos::rcp(
            new VertexCFD::ComputeErrorNorms::ErrorNorms<double>(
                mesh,
                linear_object_factory,
                error_norm_response_library,
                physics_blocks,
                *cm_factory,
                *closure_params,
                *user_params,
                eq_set_factory,
                volume_value,
                integration_order));

        // Set error norm output observer.
        auto tempus_error_norm_observer = Teuchos::rcp(
            new VertexCFD::TempusObserver::ErrorNormOutput<double>(
                error_norm_list, comp_error_norms));
        integrator_observer->addObserver(tempus_error_norm_observer);
    }

    // Set iteration output observer.
    auto tempus_iteration_observer = Teuchos::rcp(
        new VertexCFD::TempusObserver::IterationOutput<double>(dt_strategy));
    integrator_observer->addObserver(tempus_iteration_observer);

    // Set response output observer.
    if (Teuchos::nonnull(response_output_params))
    {
        auto tempus_response_observer = Teuchos::rcp(
            new VertexCFD::TempusObserver::ResponseOutput<double>(
                responses, response_output_freq));
        integrator_observer->addObserver(tempus_response_observer);
    }

    // Setup restart observer.
    if (Teuchos::nonnull(write_restart_params))
    {
        bool write_restart = false;
        if (write_restart_params->isType<bool>("Write Restart"))
        {
            write_restart = write_restart_params->get<bool>("Write Restart");
        }
        if (write_restart)
        {
            auto restart_writer
                = Teuchos::rcp(new VertexCFD::Mesh::RestartWriter(
                    mesh, dof_manager, *write_restart_params));
            auto restart_observer = Teuchos::rcp(
                new VertexCFD::TempusObserver::WriteRestart<double>(
                    restart_writer, *write_restart_params));
            integrator_observer->addObserver(restart_observer);
        }
    }

    // Set up matrix write observer
    if (Teuchos::nonnull(write_matrix_params))
    {
        bool write_matrix = false;
        if (write_matrix_params->isType<bool>("Write Matrix"))
        {
            write_matrix = write_matrix_params->get<bool>("Write Matrix");
        }
        if (write_matrix)
        {
            auto tempus_writematrix_observer = Teuchos::rcp(
                new VertexCFD::TempusObserver::WriteMatrix<double>(
                    *write_matrix_params));
            integrator_observer->addObserver(tempus_writematrix_observer);
        }
    }

    // Give our composite observer to the integrator.
    integrator->setObserver(integrator_observer);

    // Initialize integrator.
    integrator->initialize();

    // Setup time step control.
    auto tsc = integrator->getNonConstTimeStepControl();
    tsc->setTimeStepControlStrategy(dt_strategy);

    // Initialize solution.
    integrator->initializeSolutionHistory(t_init, solution, solution_dot);

    // Solve.
    integrator->advanceTime();

    // Output timing.
    if (use_timers)
    {
        stacked_timer->stop("Physics");
        Teuchos::StackedTimer::OutputOptions options;
        options.output_fraction = true;
        options.output_total_updates = true;
        options.output_minmax = true;
        options.output_proc_minmax = true;
        options.align_columns = true;
        options.print_names_before_values = true;
        options.output_histogram = false;
        options.max_levels = 100;
        options.drop_time = -1.0;
        if (profiling_params->isType<int>("Max Output Levels"))
        {
            options.max_levels
                = profiling_params->get<int>("Max Output Levels");
        }
        if (profiling_params->isType<double>("Minimum Cutoff"))
        {
            options.drop_time
                = profiling_params->get<double>("Minimum Cutoff");
        }
        stacked_timer->report(std::cout, comm, options);
    }
}

//---------------------------------------------------------------------------//
// Main function
int main(int argc, char* argv[])
{
    // Start MPI. Panzer and other Trilinos components want the Teuchos MPI
    // environment initialized so we need to do this here. We can still get
    // raw MPI communicators as needed.
    Teuchos::GlobalMPISession mpi_session(&argc, &argv, nullptr);

    // Kokkos scopeguard (initialize and finalize)
    Kokkos::ScopeGuard kokkos(argc, argv);

    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input file and initialize mesh manager
    Teuchos::RCP<VertexCFD::Parameter::ParameterDatabase> parameter_db;
    Teuchos::RCP<VertexCFD::MeshManager> mesh_manager;
    parse_input_file_and_mesh(argc, argv, comm, parameter_db, mesh_manager);

    // Get mesh dimension from mesh manager
    const int mesh_dimension = mesh_manager->mesh()->getDimension();

    // Run VertexCFD
    switch (mesh_dimension)
    {
        case 2:
            run_vertexcfd<2>(comm, parameter_db, mesh_manager);
            break;
        case 3:
            run_vertexcfd<3>(comm, parameter_db, mesh_manager);
            break;
        default:
            const std::string msg =
        "\n\nERROR:\n"
        "The mesh dimension read from the Exodus mesh file\n"
        "or the inline mesh is found to be '" +
        std::to_string(mesh_dimension) +
        "'\n. Only 2D and 3D meshes are currently supported.\n"
        "Please check your input file.\n";
            throw std::runtime_error(msg);
    }

    return 0;
}

//---------------------------------------------------------------------------//
