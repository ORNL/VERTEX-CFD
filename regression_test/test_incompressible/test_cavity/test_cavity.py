import sys
import os
import pytest

import vertexcfd_test


class TestCavity:
    # setup method for class
    @classmethod
    def setup_class(self):
        """ setup any state specific to the execution of the
          given class (which usually contains tests)."""
        print('\n***************** Setup class method *******************')

        # Class variables (to not change)
        self.mesh_file_path = vertexcfd_test.mesh_file_path
        self.input_file_path = vertexcfd_test.input_file_path
        self.vertexcfd_exec = vertexcfd_test.vertexcfd_exec
        self.exodiff_exec = vertexcfd_test.exodiff_exec
        self.main_path = vertexcfd_test.main_path
        self.no_exodiff_file = None

        # Change working directory to test directory
        self.test_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.test_path)

    # teardown method for class
    @classmethod
    def teardown_class(self):
        """ teardown any state that was previously
          setup with a call to setup_class."""
        print('\n****************** Teardown class method ******************')
        # Change working directory back to main directory
        os.chdir(vertexcfd_test.main_path)

    # setup method for each function
    def setup_method(self):
        """ setup test."""
        print('\nSetup method')
        # Initialize input file, mesh file and output file names
        self.input_file = None
        self.mesh_file = None
        self.output_file = None

    # teardown method for each function
    def teardown_method(self):
        """ teardown test."""
        print('\nTeardown method')
        # Delete input file, mesh file and output file
        vertexcfd_test.clean_working_directory(self)

    #############################################################################
    #############################################################################
    #############################################################################
    @pytest.mark.push
    @pytest.mark.daily
    @pytest.mark.weekly
    @pytest.mark.incompressible
    def test_2d_triangular_cavity(self, request):
        # Parameters
        mark_expr = request.config.option.markexpr
        vertexcfd_options = ()
        copy_xml_file = False

        # default parameters to be used on 'push'
        base_input_file = "incompressible_2d_triangular_cavity"
        self.input_file = base_input_file + "_Re100.xml"
        source_input_file = self.input_file
        output_name = base_input_file + "_Re100_solution.exo"
        nu = 0.8

        if ("daily" in mark_expr):
            self.input_file = base_input_file + "_Re400.xml"
            output_name = base_input_file + "_Re400_solution.exo"
            nu = 0.2
        elif ("weekly" in mark_expr):
            self.input_file = base_input_file + "_Re800.xml"
            output_name = base_input_file + "_Re800_solution.exo"
            nu = 0.1

        xml_args_to_replace_list = []
        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Kinematic viscosity', nu))
        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Exodus Output File',
                                               output_name))

        vertexcfd_test.create_xml_file_from_base_file(
            self, source_input_file, xml_args_to_replace_list)

        # Parameters for exodiff executables
        exodiff_options = ()

        #### The following code should not be modified ####
        # Run VertexCFD and call exodiff to compare to gold file
        vertexcfd_test.run_test(self, vertexcfd_options, exodiff_options,
                                copy_xml_file)
