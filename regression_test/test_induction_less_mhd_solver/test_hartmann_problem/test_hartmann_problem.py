import sys
import os
import pytest

import vertexcfd_test


class TestHartmannProblem:
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
    @pytest.mark.weekly
    @pytest.mark.mhd
    @pytest.mark.gpu
    def test_2d_hartmann_problem(self, request):
        #### Parameters to edit for each function ####
        # CPU or GPU
        run_gpu = False
        markers = request.config.option.markexpr.split()
        if ('gpu' in markers):
            run_gpu = True

        # Set polynomial order based on marker: p = 1 on push and gpu, and p = 2 on weekly
        p = 1
        ha = 1
        if ('gpu' in markers and 'push' in markers):
            p = 2
            ha = 1
        elif ('gpu' in markers and 'weekly' in markers):
            p = 1
            ha = 100
        elif ('weekly' in markers):
            p = 2
            ha = 100

        # Parameters for VertexCFD run
        vertexcfd_options = ()

        # Input file and output file
        copy_xml_file = False
        base_input_file = "mhd_2d_hartmann_pb_periodic_insulating"
        if run_gpu:
            base_input_file += "_cuda"
        source_input_file = base_input_file + ".xml"
        name = "_re_100_ha_" + str(ha) + "_p_" + str(p)
        base_input_file = source_input_file.split(".xml")[0]
        self.input_file = base_input_file + name + ".xml"
        output_name = base_input_file + name + "_solution.exo"

        # Set parameter values
        momentum_source = None
        ext_magn_field = None
        nx = None
        ny = None
        if ha == 1:
            momentum_source = "{0.04194528049, 0.0}"
            ext_magn_field = "{0.0, 0.1, 0.0}"
            nx = 8
            ny = 32
        elif ha == 100:
            momentum_source = "{101.010101, 0.0}"
            ext_magn_field = "{0.0, 10.0, 0.0}"
            nx = 64
            ny = 256

        # Update parameters in input file
        xml_args_to_replace_list = []

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Exodus Output File',
                                               output_name))

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Momentum Source',
                                               momentum_source))

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace(
                'Parameter', 'name', 'External Magnetic Field Value',
                ext_magn_field))

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'X Elements', nx))

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Y Elements', ny))

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Basis Order', p))

        xml_args_to_replace_list.append(
            vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                               'Integration Order', 2 * p))

        vertexcfd_test.create_xml_file_from_base_file(self,
                                                      source_input_file,
                                                      xml_args_to_replace_list,
                                                      restart=False)

        # Parameters for exodiff executables
        exodiff_options = ()

        #### The following code should not be modified ####
        # Run VertexCFD and call exodiff to compare to gold file
        vertexcfd_test.run_test(self,
                                vertexcfd_options,
                                exodiff_options,
                                copy_xml_file,
                                run_gpu=run_gpu)
