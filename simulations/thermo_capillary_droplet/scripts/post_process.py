#!/usr/bin/env python3

from copy import copy
import pyvista as pv
import numpy as np
import sys
import os

characteristic_velocity = 0.024
characteristic_time = 0.06


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: {} <input>".format(sys.argv[0]))
        print("  <input> input data, either a .pvd file or a directory containing one .pvd file")
        print("Runs postprocessing relevant for thermo capillary droplet benchmark.")
        print("The result will be ")
        sys.exit(0)

    src = sys.argv[1]
    assert os.path.exists(
        src), 'The specified input file/directory doesn\'t exist! Abort...'
    if os.path.isdir(src):
        ddir = copy(src)
        files = [filename for filename in os.listdir(
            ddir) if filename.endswith(".pvd")]
        assert len(files) > 0, 'Cannot find paraview data file in directory:\n' + \
            os.path.abspath(ddir) + '\nAbort!...'
        assert len(files) < 2, 'Multiple paraview data files in directory:\n' + os.path.abspath(ddir) + \
            '\nPlease specify one data file using its path!\nAbort!...'
        src = os.path.join(ddir, files[0])
    else:
        assert src.endswith('.pvd'), 'Unknown file type of src file! Abort...'
        ddir = os.path.dirname(src)

    requested_fields = ["level_set", "velocity"]

    pvd_reader = pv.get_reader(src)

    time_vector = np.array(pvd_reader.time_values)
    dimless_time_vector = time_vector / characteristic_time
    centroid_y_vector = np.zeros(time_vector.shape)
    max_vel_vector = np.zeros(time_vector.shape)
    avg_vel_vector = np.zeros(time_vector.shape)

    for i, time in enumerate(time_vector):
        print("processing time step {}/{} at simulation time {:.4e}".format(i,
              len(time_vector), time), end='\r')
        # read timestep
        pvd_reader.set_active_time_value(time)
        vtu_reader = pvd_reader.active_readers[0]
        vtu_reader.disable_all_cell_arrays()
        all_fields = vtu_reader.point_array_names
        vtu_reader.disable_all_point_arrays()
        for field in requested_fields:
            assert field in all_fields, "the requested field \"" + \
                field + "\" is not in the dataset!"
            vtu_reader.enable_point_array(field)
        mesh = vtu_reader.read()
        # Determine whether the droplet is the positive or the negative level set phase by sampling the level set at the
        # centre of the domain. This assumes that the droplet is initially at the centre of the domain.
        if i == 0:
            gas_inside = pv.PolyData(mesh.center).sample(mesh)[
                "level_set"][0] < 0
        droplet = mesh.clip_scalar(
            scalars="level_set", value=0, invert=gas_inside)
        # compute output
        droplet.point_data.remove("level_set")
        droplet["y"] = droplet.points[:, 1]
        droplet["vel_y"] = droplet["velocity"][:, 1]
        droplet.point_data.remove("velocity")
        max_vel_vector[i] = np.max(droplet["vel_y"])
        integrated_drop = droplet.integrate_data()
        centroid_y_vector[i] = integrated_drop["y"][0] / droplet.area
        avg_vel_vector[i] = integrated_drop["vel_y"][0] / droplet.area
    print("processing time step {}/{} at simulation time {:.4e}".format(
        len(time_vector), len(time_vector), time))
    vel_vector = np.gradient(centroid_y_vector, time_vector)
    dimless_vel_vector = vel_vector / characteristic_velocity
    dimless_max_vel_vector = max_vel_vector / characteristic_velocity
    dimless_avg_vel_vector = avg_vel_vector / characteristic_velocity

    # save vectors to csv
    output_filename = (os.path.basename(sys.argv[1][:-1] if sys.argv[1].endswith(
        "/") else sys.argv[1]) if os.path.join(sys.argv[1]) != "." else os.path.basename(src)) + ".csv"
    np.savetxt(output_filename,
               np.stack((
                   time_vector,
                   centroid_y_vector,
                   dimless_time_vector,
                   dimless_vel_vector,
                   dimless_max_vel_vector,
                   dimless_avg_vel_vector
               ), axis=1),
               delimiter=",",
               comments='',
               header="time,y_center,t/tr,u/ur,u_max/ur,u_avg/ur")
