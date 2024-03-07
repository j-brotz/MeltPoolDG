# trace generated using paraview version 5.9.0-RC3

# import the simple module from the paraview
from paraview.simple import *
import os
import argparse
from utils import find_filenames

import re
import pandas as pd
import numpy as np


def process_pvd(folder, pvd_file, isosurface_level=0.0):
    # disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    solutionpvd = PVDReader(registrationName=pvd_file,
                            FileName=os.path.join(folder, pvd_file))
    solutionpvd.PointArrays = ['velocity', 'force_rhs_velocity', 'level_set', 'normal_0', 'normal_1',
                               'curvature', 'heaviside', 'distance', 'pressure', 'force_rhs_pressure', 'density', 'viscosity']

    # get animation scene
    animationScene1 = GetAnimationScene()

    # update animation scene based on data timesteps
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    solutionpvdDisplay = Show(solutionpvd, renderView1,
                              'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    solutionpvdDisplay.Representation = 'Surface'
    solutionpvdDisplay.ColorArrayName = [None, '']
    solutionpvdDisplay.SelectTCoordArray = 'None'
    solutionpvdDisplay.SelectNormalArray = 'None'
    solutionpvdDisplay.SelectTangentArray = 'None'
    solutionpvdDisplay.OSPRayScaleArray = 'curvature'
    solutionpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    solutionpvdDisplay.SelectOrientationVectors = 'None'
    solutionpvdDisplay.ScaleFactor = 3.9999998989515007e-05
    solutionpvdDisplay.SelectScaleArray = 'None'
    solutionpvdDisplay.GlyphType = 'Arrow'
    solutionpvdDisplay.GlyphTableIndexArray = 'None'
    solutionpvdDisplay.GaussianRadius = 1.9999999494757505e-06
    solutionpvdDisplay.SetScaleArray = ['POINTS', 'curvature']
    solutionpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    solutionpvdDisplay.OpacityArray = ['POINTS', 'curvature']
    solutionpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    solutionpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    solutionpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    solutionpvdDisplay.ScalarOpacityUnitDistance = 1.4030775249419954e-05
    solutionpvdDisplay.OpacityArrayName = ['POINTS', 'curvature']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    solutionpvdDisplay.ScaleTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 33195.54296875, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    solutionpvdDisplay.OpacityTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 33195.54296875, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera()

    # changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.0, 0.0, 10000.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Contour'
    contour1 = Contour(registrationName='Contour1', Input=solutionpvd)
    contour1.ContourBy = ['POINTS', 'curvature']
    contour1.Isosurfaces = [16597.771484375]
    contour1.PointMergeMethod = 'Uniform Binning'

    # Properties modified on contour1
    contour1.ContourBy = ['POINTS', 'level_set']
    contour1.Isosurfaces = [isosurface_level]

    # show data in view
    contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'level_set'
    level_setLUT = GetColorTransferFunction('level_set')

    # trace defaults for the display properties.
    contour1Display.Representation = 'Surface'
    contour1Display.ColorArrayName = ['POINTS', 'level_set']
    contour1Display.LookupTable = level_setLUT
    contour1Display.SelectTCoordArray = 'None'
    contour1Display.SelectNormalArray = 'None'
    contour1Display.SelectTangentArray = 'None'
    contour1Display.OSPRayScaleArray = 'level_set'
    contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    contour1Display.SelectOrientationVectors = 'None'
    contour1Display.ScaleFactor = 3.0001800041645765e-05
    contour1Display.SelectScaleArray = 'level_set'
    contour1Display.GlyphType = 'Arrow'
    contour1Display.GlyphTableIndexArray = 'level_set'
    contour1Display.GaussianRadius = 1.5000900020822884e-06
    contour1Display.SetScaleArray = ['POINTS', 'level_set']
    contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
    contour1Display.OpacityArray = ['POINTS', 'level_set']
    contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
    contour1Display.DataAxesGrid = 'GridAxesRepresentation'
    contour1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    contour1Display.ScaleTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    contour1Display.OpacityTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(solutionpvd, renderView1)

    # show color bar/color legend
    contour1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get opacity transfer function/opacity map for 'level_set'
    level_setPWF = GetOpacityTransferFunction('level_set')

    # create a new 'Integrate Variables'
    integrateVariables1 = IntegrateVariables(
        registrationName='IntegrateVariables1', Input=contour1)

    # Create a new 'SpreadSheet View'
    spreadSheetView1 = CreateView('SpreadSheetView')
    spreadSheetView1.ColumnToSort = ''
    spreadSheetView1.BlockSize = 1024

    # show data in view
    integrateVariables1Display = Show(
        integrateVariables1, spreadSheetView1, 'SpreadSheetRepresentation')

    # get layout
    layout1 = GetLayoutByName("Layout #1")

    # add view to a layout so it's visible in UI
    AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=0)

    SelectIDs(IDs=[-1, 0], FieldType=1, ContainingCells=0)

    # set active source
    SetActiveSource(integrateVariables1)

    # set active source
    SetActiveSource(contour1)

    # show data in view
    contour1Display_1 = Show(contour1, spreadSheetView1,
                             'SpreadSheetRepresentation')

    # save data
    SaveData(os.path.join(folder, pvd_file.split(".")[0] + '_contour.csv'), proxy=contour1, WriteTimeSteps=1,
             PointDataArrays=['curvature', 'density', 'distance', 'force_rhs_pressure', 'force_rhs_velocity',
                              'heaviside', 'level_set', 'normal_0', 'normal_1', 'pressure', 'velocity', 'viscosity'],
             AddTime=1)

    # ================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    # ================================================================

    # --------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(953, 740)

    # -----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.0, 0.0, 10000.0]
    renderView1.CameraParallelScale = 0.0002828427053294111

    # --------------------------------------------
    # uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).

# -------------------------------------------------------
# [MS] modifications
# -------------------------------------------------------


def export_amplitude_over_time(folder, pvd_file, filename_suffix=".csv"):

    csvs = find_filenames(folder, filename_suffix,
                          pvd_file.split(".")[0] + "_contour")
    csvs.sort(key=lambda f: int(re.sub('\\D', '', f)))

    x_max = []
    y_max = []
    time_list = []

    for csv in csvs:
        data = pd.read_csv(os.path.join(folder, csv))

        x = pd.DataFrame(data, columns=['Points:0']).to_numpy()
        y = pd.DataFrame(data, columns=['Points:1']).to_numpy()
        time = pd.DataFrame(data, columns=['Time']).to_numpy()

        indices_zero = np.where(y == 0)[0]
        x_max.append(float(max(x[indices_zero])))
        indices_zero = np.where(x == 0)[0]
        y_max.append(float(max(y[indices_zero])))
        time_list.append(time[0])

    np.savetxt(os.path.join(folder, pvd_file.split(".")[0] + "_amplitude_over_time.csv"), np.column_stack((np.asarray(
        time_list), np.asarray(x_max), np.asarray(y_max))), delimiter=",", header="time, x_semi_axis, y_semi_axis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export the level set contour to a file. Execute with pvpython!')
    parser.add_argument('--folder', type=str,
                        help='define the folder, where the existing pvd-file is located')
    parser.add_argument('--pvdfile', type=str, required=False,
                        help='define the name of the processed pvd-file, e.g. solution.pvd')
    parser.add_argument('--contourlevel', type=float, required=False, default=0.0,
                        help='define the contour level of the level set function within [-1, 1]')
    args = parser.parse_args()
    print(args)

    folder = args.folder
    folder = os.path.join(os.getcwd(), folder)
    pvdfile = args.pvdfile
    contourlevel = args.contourlevel
    assert contourlevel >= -1 and contourlevel <= 1

    if not pvdfile:
        pvdfile = find_filenames(folder)
        assert len(pvdfile) == 1
        pvdfile = pvdfile[0]

    print(70 * "-")
    print(
        " Start processing pvd-file: {:}".format(os.path.join(folder, pvdfile)))
    process_pvd(folder, pvdfile, contourlevel)
    export_amplitude_over_time(folder, pvdfile)
    print(" The end")
    print(70 * "-")
# -------------------------------------------------------
# [MS] modifications end
# -------------------------------------------------------
