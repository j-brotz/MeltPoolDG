# trace generated using paraview version 5.9.0-RC3

# import the simple module from the paraview
from paraview.simple import *
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from utils import find_filenames


def process_pvd(folder, pvd_file):
    # disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'

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
    solutionpvdDisplay.ScalarOpacityUnitDistance = 2.2272467390858614e-05
    solutionpvdDisplay.OpacityArrayName = ['POINTS', 'curvature']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    solutionpvdDisplay.ScaleTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 31830.794921875, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    solutionpvdDisplay.OpacityTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 31830.794921875, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera()

    # changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.0, 0.0, 10000.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator1', Input=solutionpvd)
    calculator1.Function = ''

    # Properties modified on calculator1
    calculator1.ResultArrayName = 'KineticEnergy'
    calculator1.Function = 'density*velocity.velocity'

    # show data in view
    calculator1Display = Show(calculator1, renderView1,
                              'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'KineticEnergy'
    kineticEnergyLUT = GetColorTransferFunction('KineticEnergy')

    # get opacity transfer function/opacity map for 'KineticEnergy'
    kineticEnergyPWF = GetOpacityTransferFunction('KineticEnergy')

    # trace defaults for the display properties.
    calculator1Display.Representation = 'Surface'
    calculator1Display.ColorArrayName = ['POINTS', 'KineticEnergy']
    calculator1Display.LookupTable = kineticEnergyLUT
    calculator1Display.SelectTCoordArray = 'None'
    calculator1Display.SelectNormalArray = 'None'
    calculator1Display.SelectTangentArray = 'None'
    calculator1Display.OSPRayScaleArray = 'KineticEnergy'
    calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    calculator1Display.SelectOrientationVectors = 'None'
    calculator1Display.ScaleFactor = 3.9999998989515007e-05
    calculator1Display.SelectScaleArray = 'KineticEnergy'
    calculator1Display.GlyphType = 'Arrow'
    calculator1Display.GlyphTableIndexArray = 'KineticEnergy'
    calculator1Display.GaussianRadius = 1.9999999494757505e-06
    calculator1Display.SetScaleArray = ['POINTS', 'KineticEnergy']
    calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
    calculator1Display.OpacityArray = ['POINTS', 'KineticEnergy']
    calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
    calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
    calculator1Display.PolarAxes = 'PolarAxesRepresentation'
    calculator1Display.ScalarOpacityFunction = kineticEnergyPWF
    calculator1Display.ScalarOpacityUnitDistance = 2.2272467390858614e-05
    calculator1Display.OpacityArrayName = ['POINTS', 'KineticEnergy']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    calculator1Display.ScaleTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    calculator1Display.OpacityTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(solutionpvd, renderView1)

    # show color bar/color legend
    calculator1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Integrate Variables'
    integrateVariables1 = IntegrateVariables(
        registrationName='IntegrateVariables1', Input=calculator1)

    # set active source
    SetActiveSource(integrateVariables1)

    # show data in view
    integrateVariables1Display = Show(
        integrateVariables1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    integrateVariables1Display.Representation = 'Surface'
    integrateVariables1Display.ColorArrayName = ['POINTS', 'KineticEnergy']
    integrateVariables1Display.LookupTable = kineticEnergyLUT
    integrateVariables1Display.SelectTCoordArray = 'None'
    integrateVariables1Display.SelectNormalArray = 'None'
    integrateVariables1Display.SelectTangentArray = 'None'
    integrateVariables1Display.OSPRayScaleArray = 'KineticEnergy'
    integrateVariables1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    integrateVariables1Display.SelectOrientationVectors = 'None'
    integrateVariables1Display.ScaleFactor = 0.1
    integrateVariables1Display.SelectScaleArray = 'KineticEnergy'
    integrateVariables1Display.GlyphType = 'Arrow'
    integrateVariables1Display.GlyphTableIndexArray = 'KineticEnergy'
    integrateVariables1Display.GaussianRadius = 0.005
    integrateVariables1Display.SetScaleArray = ['POINTS', 'KineticEnergy']
    integrateVariables1Display.ScaleTransferFunction = 'PiecewiseFunction'
    integrateVariables1Display.OpacityArray = ['POINTS', 'KineticEnergy']
    integrateVariables1Display.OpacityTransferFunction = 'PiecewiseFunction'
    integrateVariables1Display.DataAxesGrid = 'GridAxesRepresentation'
    integrateVariables1Display.PolarAxes = 'PolarAxesRepresentation'
    integrateVariables1Display.ScalarOpacityFunction = kineticEnergyPWF
    integrateVariables1Display.ScalarOpacityUnitDistance = 0.0
    integrateVariables1Display.OpacityArrayName = ['POINTS', 'KineticEnergy']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    integrateVariables1Display.ScaleTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    integrateVariables1Display.OpacityTransferFunction.Points = [
        0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # show color bar/color legend
    integrateVariables1Display.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(calculator1, renderView1)

    # get layout
    layout1 = GetLayout()

    # split cell
    layout1.SplitHorizontal(0, 0.5)

    # set active view
    SetActiveView(None)

    # Create a new 'SpreadSheet View'
    spreadSheetView1 = CreateView('SpreadSheetView')
    spreadSheetView1.ColumnToSort = ''
    spreadSheetView1.BlockSize = 1024

    # assign view to a particular cell in the layout
    AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=2)

    # show data in view
    integrateVariables1Display_1 = Show(
        integrateVariables1, spreadSheetView1, 'SpreadSheetRepresentation')

    animationScene1.GoToNext()

    animationScene1.GoToNext()

    animationScene1.GoToFirst()

    # save data
    SaveData(os.path.join(folder, pvd_file.split(".")[0] + '_kinetic_energy.csv'), proxy=integrateVariables1, WriteTimeSteps=1,
             ChooseArraysToWrite=1,
             PointDataArrays=['KineticEnergy'],
             AddMetaData=0,
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
def collect_files_and_write_into_one(folder, pvdfile, suffix=".csv"):

    if (os.path.isfile(os.path.join(folder, pvdfile.split(".")[0] + "_kinetic_energy.csv"))):
        os.remove(os.path.join(folder, pvdfile.split(
            ".")[0] + "_kinetic_energy.csv"))
    csvs = find_filenames(folder, suffix, pvdfile.split(".")[
                          0] + "_kinetic_energy")
    csvs.sort(key=lambda f: int(re.sub('\\D', '', f)))

    kinetic_energy = []
    time_list = []

    for csv in csvs:

        data = pd.read_csv(os.path.join(folder, csv))

        kinetic_energy.append(
            float(pd.DataFrame(data, columns=['KineticEnergy']).to_numpy()))
        time_list.append(
            float(pd.DataFrame(data, columns=['Time']).to_numpy()[-1]))
        os.remove(os.path.join(folder, csv))

    np.savetxt(os.path.join(folder, pvdfile.split(".")[0] + "_kinetic_energy.csv"), np.column_stack((np.asarray(
        time_list), np.asarray(kinetic_energy))), delimiter=",", header="time, kinetic energy")
    print(" file written: {:}".format(
        os.path.join(folder, pvdfile.split(".")[0] + "_kinetic_energy.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export the level set contour to a file. Execute with pvpython!')
    parser.add_argument('--folder', type=str,
                        help='define the folder, where the existing pvd-file is located')
    parser.add_argument('--pvdfile', type=str, required=False,
                        help='define the name of the processed pvd-file, e.g. solution.pvd')
    args = parser.parse_args()
    print(args)

    folder = args.folder
    folder = os.path.join(os.getcwd(), folder)
    pvdfile = args.pvdfile

    if not pvdfile:
        pvdfile = find_filenames(folder)
        assert len(pvdfile) == 1
        pvdfile = pvdfile[0]

    print(70 * "-")
    print(
        " Start processing pvd-file: {:}".format(os.path.join(folder, pvdfile)))
    process_pvd(folder, pvdfile)
    collect_files_and_write_into_one(folder, pvdfile)
    print(" The end")
    print(70 * "-")
# -------------------------------------------------------
# [MS] modifications end
# -------------------------------------------------------
