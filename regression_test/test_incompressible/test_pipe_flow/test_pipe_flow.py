import sys
import os
import pytest
import re

import vertexcfd_test


class TestPipeFlow:
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

    # Check inlet conditions (velocity and temperature)
    def check_inlet_conditions(self, capfd):
        # Get captured test output for parsing
        fd_out, fd_err = capfd.readouterr()

        # Let pytest re-capture the output
        sys.stdout.write(fd_out)
        sys.stderr.write(fd_err)

        pattern = re.compile('  Inlet - velocity_0 = (.+)\n')
        vels = [float(t) for t in pattern.findall(fd_out)]
        pattern = re.compile('  Inlet - temperature = (.+)\n')
        temps = [float(t) for t in pattern.findall(fd_out)]

        # Expected velocity and temperature values
        vel_exp = [0.000600000000, 0.000600000000]
        temp_exp = [4.0, 4.047127939785872]

        # Assert values
        assert len(vels) == 2
        assert len(temps) == 2

        for i in range(0, 2):
            assert vels[i] == pytest.approx(vel_exp[i], rel=1.0e-8)
            assert temps[i] == pytest.approx(temp_exp[i], rel=1.0e-8)

    # Check probe values
    def check_probe_values(self, capfd):
        # Get captured test output for parsing
        fd_out, fd_err = capfd.readouterr()

        # Let pytest re-capture the output
        sys.stdout.write(fd_out)
        sys.stderr.write(fd_err)

        pattern = re.compile('  Probe Upper 1 - temperature = (.+)\n')
        pb1 = [float(t) for t in pattern.findall(fd_out)]
        pattern = re.compile('  Probe Upper 2 - temperature = (.+)\n')
        pb2 = [float(t) for t in pattern.findall(fd_out)]
        pattern = re.compile('  Probe Right 1 - temperature = (.+)\n')
        pb3 = [float(t) for t in pattern.findall(fd_out)]

        # Expected velocity and temperature values
        pb1_exp = [20.0, 20.65956051875808]
        pb2_exp = [20.0, 20.03234994314487]
        pb3_exp = [
            20.0, 20.00184327050597, 20.00184327047455, 20.00184327047455
        ]

        # Assert values
        assert len(pb1) == 2
        assert len(pb2) == 2
        assert len(pb3) == 4

        for i in range(0, 2):
            assert pb1[i] == pytest.approx(pb1_exp[i], rel=1.0e-8)
            assert pb2[i] == pytest.approx(pb2_exp[i], rel=1.0e-8)

        for i in range(0, 4):
            assert pb3[i] == pytest.approx(pb3_exp[i], rel=1.0e-8)

    @pytest.mark.push
    @pytest.mark.incompressible
    def test_2d_flow(self):
        #### Parameters to edit for each function ####
        # Parameters for VertexCFD run
        self.input_file = "incompressible_2d_channel_periodic.xml"
        vertexcfd_options = ()

        # Parameters for exodiff executables
        exodiff_options = ()

        #### The following code should not be modified ####
        # Run VertexCFD and call exodiff to compare to gold file
        vertexcfd_test.run_test(self, vertexcfd_options, exodiff_options)

    @pytest.mark.push
    @pytest.mark.incompressible
    def test_2d_heated_flow(self, capfd):
        #### Parameters to edit for each function ####
        # Parameters for VertexCFD run
        self.input_file = "incompressible_2d_heated_channel.xml"
        vertexcfd_options = ()

        # Parameters for exodiff executables
        exodiff_options = ()

        #### The following code should not be modified ####
        # Run VertexCFD and call exodiff to compare to gold file
        vertexcfd_test.run_test(self, vertexcfd_options, exodiff_options)

        # Check surface-averaged values
        self.check_inlet_conditions(capfd)

        # Check probe values
        self.check_probe_values(capfd)

    @pytest.mark.daily
    @pytest.mark.incompressible
    def test_3d_flow(self):
        #### Parameters to edit for each function ####
        # Parameters for VertexCFD run
        self.input_file = "incompressible_3d_channel_periodic.xml"
        vertexcfd_options = ()

        # Parameters for exodiff executables
        exodiff_options = ()

        #### The following code should not be modified ####
        # Run VertexCFD and call exodiff to compare to gold file
        vertexcfd_test.run_test(self, vertexcfd_options, exodiff_options)
