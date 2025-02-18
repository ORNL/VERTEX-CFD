import sys
import os
import pytest

import vertexcfd_test


class TestTaylorGreenVortex:
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

        # Gold values for L1/L2 error norms
        self.ref_error_norms = [[
            None,
            [
                3.1354940494154192e-03, 4.8827471008265188e-03,
                4.8827471008264806e-03
            ],
            [
                6.1898331388840358e-04, 9.5931336430924428e-04,
                9.5931336430924038e-04
            ]
        ],
                                [
                                    None,
                                    [
                                        7.2363659991042098e-04,
                                        1.0898079401035921e-03,
                                        1.0898079401035173e-03
                                    ],
                                    [
                                        1.4131616203283908e-04,
                                        2.1968921759816913e-04,
                                        2.1968921759815726e-04
                                    ]
                                ],
                                [
                                    None,
                                    [
                                        1.1253018483846534e-04,
                                        1.8455771918065726e-04,
                                        1.8455771918063306e-04
                                    ],
                                    [
                                        2.2227805982151793e-05,
                                        4.0783845770510348e-05,
                                        4.0783845770502603e-05
                                    ]
                                ]]

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
        #vertexcfd_test.clean_working_directory(self)

    #############################################################################
    #############################################################################
    #############################################################################
    @pytest.mark.push
    @pytest.mark.daily
    @pytest.mark.weekly
    @pytest.mark.incompressible
    def test_2d_mesh_convergence_laminar(self, request, capfd):
        #### Parameters to edit for each function ####
        # Parameters for VertexCFD run
        copy_xml_file = False
        restart = False
        exodiff_options = None
        base_input_file = "incompressible_2d_taylor_green_vortex"
        final_time = 50
        source_input_file = base_input_file + ".xml"

        # Parameters for mesh convergence study
        mark_expr = request.config.option.markexpr
        xy_list = [20]
        if (mark_expr == 'weekly'):
            xy_list = [20, 40]
        elif (mark_expr == 'weekly'):
            xy_list = [20, 40, 80]
        errors = []

        # Loop over all elements of 'xy_list'
        for i in range(0, len(xy_list)):
            xy = xy_list[i]
            self.input_file = base_input_file + "_" + str(
                final_time) + "_" + str(final_time) + "_" + str(xy) + ".xml"
            output_name = base_input_file + "_" + str(final_time) + "_" + str(
                xy) + "_solution.exo"

            xml_args_to_replace_list = []
            xml_args_to_replace_list.append(
                vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                                   'X Elements', xy))
            xml_args_to_replace_list.append(
                vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                                   'Y Elements', xy))
            xml_args_to_replace_list.append(
                vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                                   'Final Time', final_time))
            xml_args_to_replace_list.append(
                vertexcfd_test.xml_args_to_replace('Parameter', 'name',
                                                   'Exodus Output File',
                                                   output_name))

            vertexcfd_test.create_xml_file_from_base_file(
                self, source_input_file, xml_args_to_replace_list, restart)

            # Clear replacing parameter list
            xml_args_to_replace_list.clear()

            vertexcfd_options = ()

            # Run VertexCFD (note that there is no comparison to gold file here)
            vertexcfd_test.run_test(self, vertexcfd_options, exodiff_options,
                                    copy_xml_file)

            # Compare error norms to gold values
            vertexcfd_test.compare_error_norms(self.ref_error_norms[i], capfd,
                                               1.0e-10)
