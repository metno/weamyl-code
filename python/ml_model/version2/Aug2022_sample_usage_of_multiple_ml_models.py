import os
import shutil
import glob
import sys
import datetime
#import pyproj
import xarray as xr
#from mpl_toolkits.basemap import Basemap
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
from math import nan
from copy import deepcopy
from netCDF4 import Dataset, num2date, date2num

# httpLinkOrFile=
# httpLinkOrFile="data/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20210107.nc"
# httpLinkOrFile="data/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20220316.nc"
def loadNetCDF(
        httpLinkOrFile="data/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20220316.nc"):
    """
      Return the NetCDF Dataset
      Change the default value if you want to experiment with other netcdf files
      Can use links to threads but in this case data will be downloaded each times
      The URL for the current default value is: https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2021/01/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20210107.nc
    """
    return xr.open_dataset(httpLinkOrFile)


def loadMLModel(modelFileName):
    """
      Load Trained ML Model from the file provided by the ML Development Team form a HDF5 file
      HDF5 file - contains the model's architecture, weights as resulted after the model was trained
      Model can be loaded once and used for all subsequent predictions
    """
    import keras
    model = keras.models.load_model(modelFileName, compile=False)
    # model.summary()
    return model


def loadReflectivityAtTPoz(netCDFData, timeStampPoz):
    return netCDFData.variables["equivalent_reflectivity_factor"][timeStampPoz, :, :].data

def SaveToNetCDF4(outfilename, predicted, netCDFData, timeStamp):
    """
      Save the predicted data into a similar NetCDF file by replacing original values with predicted values
    """

    ncfile = Dataset(outfilename, mode='w', format='NETCDF4_CLASSIC')
    ncfile.title = 'Machine learning model output for WeaMyL project'
    ncfile.history = 'Running script received from UBB Romania'
    ncfile.institution = 'Norwegian Meteorological Institute (MET Norway)'
    ncfile.Conventions = "CF-1.6"
    ncfile.source = 'MET Norway'
    ncfile.reference = 'WeaMyL project'
    ncfile.contact = 'abdelkaderm@met.no'

    # create dimensions
    Yc = ncfile.createDimension('Y', 2134)
    Xc = ncfile.createDimension('X', 1694)
    time = ncfile.createDimension('time', None)

    # Create variables
    Xcs = ncfile.createVariable('Xc', np.float32, ('X',))
    Xcs[:] = netCDFData.variables['Xc'].data
    Xcs.axis = netCDFData.variables['Xc'].attrs['axis']
    Xcs.standard_name = netCDFData.variables['Xc'].attrs['standard_name']
    Xcs.units = netCDFData.variables['Xc'].attrs['units']

    Ycs = ncfile.createVariable('Yc', np.float32, ('Y',))
    Ycs[:] = netCDFData.variables['Yc'].data
    Ycs.axis = netCDFData.variables['Yc'].attrs['axis']
    Ycs.standard_name = netCDFData.variables['Yc'].attrs['standard_name']
    Ycs.units = netCDFData.variables['Yc'].attrs['units']

    lats = ncfile.createVariable('lat', np.float32, ('Y', 'X'))
    lats[:, :] = netCDFData.variables['lat'].data
    lats.units = 'degrees_north'  # netCDFData.variables['lat'].attrs['units']
    lats.long_name = netCDFData.variables['lat'].attrs['long_name']
    lats.standard_name = netCDFData.variables['lat'].attrs['standard_name']

    lons = ncfile.createVariable('lon', np.float32, ('Y', 'X'))
    lons[:, :] = netCDFData.variables['lon'].data
    lons.units = 'degrees_east'  # netCDFData.variables['lon'].attrs['units']
    lons.long_name = netCDFData.variables['lon'].attrs['long_name']
    lons.standard_name = netCDFData.variables['lon'].attrs['standard_name']

    projs = ncfile.createVariable('projection_lambert', np.int32)
    projs.standard_parallel = netCDFData.variables['projection_lambert'].attrs['standard_parallel']
    projs.proj4 = netCDFData.variables['projection_lambert'].attrs['proj4']
    projs.grid_mapping_name = netCDFData.variables['projection_lambert'].attrs['grid_mapping_name']
    projs.latitude_of_projection_origin = netCDFData.variables['projection_lambert'].attrs[
        'latitude_of_projection_origin']
    projs.longitude_of_central_meridian = netCDFData.variables['projection_lambert'].attrs[
        'longitude_of_central_meridian']
    # projs.lambert_conformal_conic: false_easting = 0.;
    # projs.lambert_conformal_conic: false_northing = 0.;

    times = ncfile.createVariable('time', 'f8', ('time',))  #
    times.axis = netCDFData.variables['time'].attrs['axis']
    times.long_name = 'time'
    times.standard_name = netCDFData.variables['time'].attrs['standard_name']
    times.units = 'seconds since 1970-01-01 00:00:00 +00:00'  # "seconds since 1800-01-01 00:00:00"
    # pdb.set_trace()
    times.calendar = 'proleptic_gregorian'

    # times[:]= date2num(np.datetime64(netCDFData.variables['time'].data[0],'s').astype(datetime.datetime),  times.units)
    # pdb.set_trace()
    times[:] = date2num(timeStamp, times.units)
    # print(date2num(timeStamp, times.units))
    # pdb.set_trace()
    rf = ncfile.createVariable('equivalent_reflectivity_factor', np.float32, ('time', 'Y', 'X'), fill_value=9.9621e+36)
    # predicted_Reflectivity[np.isnan(predicted_Reflectivity)] = 0
    #pdb.set_trace()
    rf[0, :, :] = predicted

    # np.random.random((2134,1694)) #predicted_Reflectivity
    rf.units = netCDFData.variables['equivalent_reflectivity_factor'].attrs['units']
    rf.long_name = netCDFData.variables['equivalent_reflectivity_factor'].attrs['long_name']
    rf.standard_name = netCDFData.variables['equivalent_reflectivity_factor'].attrs['standard_name']
    rf.coordinates = 'lon lat'
    rf.grid_mapping = netCDFData.variables['equivalent_reflectivity_factor'].attrs['grid_mapping']

    ncfile.close()


# def saveData(predicted_Velocity, actual_Velocity, netCDFData, hour, minute, predictionMinutes, dir,
#              secondPredicted_Velocity=None):
#     fig = createFigure(predicted_Velocity, actual_Velocity, netCDFData, hour, minute, predictionMinutes, False,
#                        secondPredicted_Velocity);
#     plt.savefig(f"{dir}testsave{hour:02d}{minute:02d}.png", bbox_inches='tight', dpi=400)
#     plt.close(fig);


# def createFigure(predicted_Velocity, actual_Velocity, netCDFData, hour, minute, predictionMinutes, showColorMap=True,
#                  secondPredicted_Velocity=None):
#     """
#     Visualize data on a map actual vs predicted side by side
#     Use Equidistant Cylindrical Projection
#     Show approximately the same area as in Godiva viewer from thredds.met.no
#     """
#     lon, lat = netCDFData.variables['lon'].data, netCDFData.variables['lat'].data
#     cmap, vmin, vmax = createColormap()
#     fig = plt.figure()
#     llcrnrlat = 52  # 43.5 #Y down
#     urcrnrlat = 75  # 82  #Y up
#     llcrnrlon = -2  # -7  #X left
#     urcrnrlon = 35  # 41  #X right
#
#     nrPlots = 2
#     if not secondPredicted_Velocity is None: nrPlots = 3
#
#     # project actual values to the map
#     subplotActual = fig.add_subplot(1, nrPlots, 1)
#     drawPrediction(subplotActual, actual_Velocity, lon, lat, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap, vmin,
#                    vmax, f"Radar at {hour:02d}:{minute:02d}")
#
#     # project predicted values to the map
#     subplotPredict = fig.add_subplot(1, nrPlots, 2)
#     im = drawPrediction(subplotPredict, predicted_Velocity, lon, lat, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap,
#                         vmin, vmax, f"Prediction ({predictionMinutes} min)")
#
#     if not secondPredicted_Velocity is None:
#         subplotPredict = fig.add_subplot(1, nrPlots, 3)
#         im = drawPrediction(subplotPredict, secondPredicted_Velocity, lon, lat, llcrnrlat, urcrnrlat, llcrnrlon,
#                             urcrnrlon, cmap, vmin, vmax, "Prediction2")
#
#     if showColorMap:
#         # show colormap
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
#         fig.colorbar(im, cax=cbar_ax)
#     return fig


# def drawPrediction(subplot, predicted_Velocity, lon, lat, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap, vmin, vmax,
#                    title):
#     subplot.set_title(title)
#     m = Basemap(llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon)
#     X, Y = m(lon, lat)
#     # draw a NASA Blue Marble image as a map background
#     m.bluemarble()
#     im = subplot.pcolormesh(X, Y, predicted_Velocity, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
#     return im


# def createColormapMet():
#     # R,G,B   Reflectivity
#     # 0.925,0.882,0.878   >53.5
#     # 0.824,0.576,0.796   39
#     # 0.788,0.220,0.149   34
#     # 0.914,0.482,0.149   29.5
#     # 0.914,0.592,0.149   23
#     # 0.914,0.698,0.149   18
#     # 0.914,0.808,0.149   14.5
#     # 0.914,0.914,0.149   7
#     # 0.753,0.753,0.329   2
#     # 0.573,0.573,0.486  -2.5
#     colors = [(0.0, (0.573, 0.573, 0.486)), (0.08035, (0.753, 0.753, 0.329)), (0.16964, (0.914, 0.914, 0.149)),
#               (0.30357, (0.914, 0.808, 0.149)), (0.36607, (0.914, 0.698, 0.149)), (0.45535, (0.914, 0.592, 0.149)),
#               (0.57142, (0.914, 0.482, 0.149)), (0.65178, (0.788, 0.22, 0.149)), (0.74107, (0.824, 0.576, 0.796)),
#               (1.0, (0.925, 0.882, 0.878))]
#     cmap_name = 'Reflectivity'
#     cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
#     vmin, vmax = -2.5, 53.5
#     return cmap, vmin, vmax
#
#
# def createColormap():
#     return createColormapMet()
#

def predictXNOW_NAN_2_Negative_1_4TInp(model, inputDataAll, replaceInputNanWith, outputNanTreashold):
    """
      inputData - 2 dimensional ndarray, the raw data in the same shape and format as in the ntcdf file
      return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    inputForTheModel1 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[0], replaceInputNanWith)
    inputForTheModel2 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[1], replaceInputNanWith)
    inputForTheModel3 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[2], replaceInputNanWith)
    inputForTheModel4 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[3], replaceInputNanWith)
    inputForTheModel = np.stack([inputForTheModel1, inputForTheModel2, inputForTheModel3, inputForTheModel4], axis=3)

    prediction = model.predict(inputForTheModel)
    # prediction = inputForTheModel
    return postProcessPredictionXNOW_NAN_2_Negative_1(prediction, outputNanTreashold)

def postProcessPredictionXNOW_NAN_2_Negative_1(prediction, outputNanTreashold):
    """
      Some post-processing of the predicted data may be necessary (crop/stitch/normalize  etc)
      For every model we will provide the description of the necessary steps and transformations
      Sample processing:
          -  stitch data predicted for smaller regions into a large region
          -  mark some values as missing (nan)
          -  divide results into multiple results (ex: predictOne2One multiple values for different timestamps in a single step)
      prediction - 2 dimensional ndarray, data as resulted form the prediction
      return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    # put nan if prediction is -2 or less
    # print("replace output <",outputNanTreashold," with nan")
    prediction[(prediction < outputNanTreashold)] = nan

    # stitch tiles together to form a single output matrix
    return combineToASingleMAtrix(prediction)

def combineToASingleMAtrix(prediction):
    """
    given a list of 2d matrix (mosaic tiles)
    reconstruct the larger matrix
    return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    # 2134, 1694
    # tiles = prediction.reshape(13, 16, 128, 128)
    # tiles = tiles.swapaxes(1, 2)
    tiles = prediction.reshape(16, 13, 128, 128)
    tiles = tiles.swapaxes(1, 2)
    tiles = tiles.reshape(2048, 1664)
    # pad with nan to the exact dimension
    result = np.empty((2134, 1694))

    # tiles = prediction.reshape(17, 16, 128, 128)
    #     tiles = tiles.swapaxes(1, 2)
    #     tiles = tiles.reshape(2176, 2048)
    #     # pad with nan to the exact dimension
    #     result = np.empty((2243, 2067))   #(2243, 2067)
    result[:] = nan
    result[:tiles.shape[0], :tiles.shape[1]] = tiles
    return result

def prepareDataForModelXNOW_NAN_2_Negative_1(productRawData, replaceInputNanWith):
    """
      Some pre-processing of the data may be necessary (crop/stitch/normalize  etc)
      For every model we will provide the description of the necessary steps and transformations
      Sample processing:
          -  crop the data to a smaller region
          -  replace missing values (nan) with a constant or an interpolated value
          -  stitch together multiple products in a single matrix
          -  stitch together product values from multiple timestamps
      productRawData - 2 dimensional ndarray, the raw data in the same shape and format as in the ntcdf file
      return 2 dimensional ndarray, data prepared for using as input for the ML Model
    """

    productRawData = deepcopy(productRawData)

    # replace negatives with -1
    productRawData[(productRawData < -1)] = -1

    # replace nan with -2
    # print("replace input nan with ",replaceInputNanWith)
    productRawData[np.isnan(productRawData)] = replaceInputNanWith

    # normalize to [0,1]
    domainMin = replaceInputNanWith
    domainMax = 75
    productRawData -= domainMin
    productRawData /= (domainMax - domainMin)

    # break into tiles
    slices = sliding_window(productRawData, 128)
    slices = slices.reshape(-1, *slices.shape[2:], 1)
    return slices

def sliding_window(a, window):
    """Very basic multi dimensional rolling window. window should be the shape of
    of the desired subarrays. Window is either a scalar or a tuple of same size
    as `arr.shape`.
    """
    h, w = a.shape
    bh, bw = window, window
    shape = (h // bh, w // bw, bh, bw)
    strides = a.itemsize * np.array([w * bh, bw, w, 1])
    blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return blocks

def predictFunc5min(x):
    """
    Predict 5 min ahead using model_1_0_0
    :param x:
    :return: predict reflectivity for the upcoming 5 minutes
    """
    predict = predictXNOW_NAN_2_Negative_1_4TInp(model1_0_0, x, -5, -2.5)
    return(predict)

def predictFunc15min(x):
    """
    Predict 15 min ahead using model_1_1_0
    :param x: input of 4 previous time steps
    :return: predict reflectivity for the upcoming 15 minutes
    """
    predict = predictXNOW_NAN_2_Negative_1_4TInp(model1_1_0, x, -5, -2.5)
    return(predict)

def predictFunc25min(x):
    """
    Predict 25 min ahead using model_1_2_0
    :param x:
    :return: predict x for the upcoming 25 minutes
    """
    predict = predictXNOW_NAN_2_Negative_1_4TInp(model1_2_0, x, -5, -2.5)
    return(predict)

def generate_multiple_3models(time, nrStepsInTheFuture):
    global input_all, t_00, acquisition_interval, predicted_Reflectivity_At_PredictionTime, \
           predicted_Reflectivity_At_PredictionTimePlus5, predicted_Reflectivity_At_PredictionTimePlus10, \
           predicted_Reflectivity_At_PredictionTimePlus15, x_25, x_20, x_10, x_05, x_15
    netCDFData = loadNetCDF()

    acquisition_interval = 5
    t_00 = time
    t_05 = (t_00 - datetime.timedelta(minutes=acquisition_interval))  # .strftime("%Y%m%dT%H%M%SZ")
    t_10 = (t_00 - datetime.timedelta(minutes=2 * acquisition_interval))  # .strftime("%Y%m%dT%H%M%SZ")
    t_15 = (t_00 - datetime.timedelta(minutes=3 * acquisition_interval))  # .strftime("%Y%m%dT%H%M%SZ")
    t_20 = (t_00 - datetime.timedelta(minutes=4 * acquisition_interval))  # .strftime("%Y%m%dT%H%M%SZ")
    t_25 = (t_00 - datetime.timedelta(minutes=5 * acquisition_interval))  # .strftime("%Y%m%dT%H%M%SZ")
    #if os.path.isfile(httpLinkOrFile_05):
    # netCDFData = loadNetCDF(httpLinkOrFile)

    radar_data_prefix = "yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000."

    if sys.argv[1] == 'latest':
        path = "/home/abdelkaderm/weamyl_tmp/"
        path = '/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/'
        http_link_or_file_05 = path + radar_data_prefix + t_05.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        netCDFData_05 = loadNetCDF(http_link_or_file_05)

        http_link_or_file_10 = path + radar_data_prefix + t_10.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        netCDFData_10 = loadNetCDF(http_link_or_file_10)

        http_link_or_file_15 = path + radar_data_prefix + t_15.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        netCDFData_15 = loadNetCDF(http_link_or_file_15)

        http_link_or_file_20 = path + radar_data_prefix + t_20.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        netCDFData_20 = loadNetCDF(http_link_or_file_20)

        http_link_or_file_25 = path + radar_data_prefix + t_25.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        netCDFData_25 = loadNetCDF(http_link_or_file_25)

        # load reflectivity at time t_**
        x_05 = loadReflectivityAtTPoz(netCDFData_05, 0)
        x_10 = loadReflectivityAtTPoz(netCDFData_10, 0)
        x_15 = loadReflectivityAtTPoz(netCDFData_15, 0)
        x_20 = loadReflectivityAtTPoz(netCDFData_20, 0)
        x_25 = loadReflectivityAtTPoz(netCDFData_25, 0)
    else:
        path = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/'
        http_link_or_file = path + time.strftime("%Y") + '/' + time.strftime("%m") + radar_data_prefix\
                            + time.strftime("%Y%m%d") + ".nc"
        netCDFData = loadNetCDF(http_link_or_file)

    #     print('Trying to open:' + httpLinkOrFile)
    #     print("Error: the radar data is not available. Please try again in few minutes !")
    #     # sys.exit(); # ('The requested forecast is not yet available. Please try again in few minutes!')

    input_all = [x_25, x_20, x_15, x_10, x_05]

    for it in range(0, nrStepsInTheFuture * acquisition_interval, 20):
        t_00 = time + datetime.timedelta(minutes=it)
        # make prediction (feed result back in until we have the desired timestep in the future
        # pdb.set_trace()
        # import pdb; pdb.set_trace()
        # Generate time t from the 4 previous time steps
        print('Generated time: ' + t_00.strftime('%H:%M') + ' .... with ML model 1.0.0')
        predicted_Reflectivity_At_PredictionTime = predictFunc5min(input_all[1:])
        #actual_Reflectivity_At_PreditionTime = loadReflectivityAtTPoz(netCDFData, timeStampPoz)

        # saveData(predicted_Reflectivity_At_PredictionTime, actual_Reflectivity_At_PreditionTime, netCDFData, hour,
        #          minute, (timeStampPoz - startTimeStampPoz + 1) * acquisition_interval, "video/", None)
        out_filename = "/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/"+ radar_data_prefix\
                       + t_00.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        SaveToNetCDF4(out_filename, predicted_Reflectivity_At_PredictionTime, netCDFData_05, t_00)

        # Generate time t+5 from the 4 previous time steps
        t05 = t_00 + datetime.timedelta(minutes=acquisition_interval)
        print("Generated time: " + t05.strftime("%H:%M") + " .... with ML model 1.1.0")
        predicted_Reflectivity_At_PredictionTimePlus5 = predictFunc15min(input_all[:-1])
        #actual_Reflectivity_At_PreditionTimePlus5 = loadReflectivityAtTPoz(netCDFData, timeStampPoz + 1)
        # saveData(predicted_Reflectivity_At_PredictionTimePlus5, actual_Reflectivity_At_PreditionTimePlus5, netCDFData,
        #          hour, minute, (timeStampPoz - startTimeStampPoz + 2) * acquisition_interval, "video/", None)

        out_filename_05 = "/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/" + radar_data_prefix \
                          + t05.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        SaveToNetCDF4(out_filename_05, predicted_Reflectivity_At_PredictionTimePlus5, netCDFData, t05)

        # Generate time t+10 from the 4 previous time steps
        t10 = t05 + datetime.timedelta(minutes=acquisition_interval)
        print("Generated time: " + t10.strftime("%H:%M") + " .... with ML model 1.1.0")
        predicted_Reflectivity_At_PredictionTimePlus10 = predictFunc15min(input_all[1:])
        #actual_Reflectivity_At_PreditionTimePlus10 = loadReflectivityAtTPoz(netCDFData, timeStampPoz + 2)
        # saveData(predicted_Reflectivity_At_PredictionTimePlus10, actual_Reflectivity_At_PreditionTimePlus10, netCDFData,
        #          hour, minute, (timeStampPoz - startTimeStampPoz + 3) * acquisition_interval, "video/", None)
        out_filename_10 = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/' + radar_data_prefix \
                          + t10.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        SaveToNetCDF4(out_filename_10, predicted_Reflectivity_At_PredictionTimePlus10, netCDFData, t10)

        # Generate time t+15 from the 4 previous time steps
        t15 = t10 + datetime.timedelta(minutes=acquisition_interval)
        print("Generated time: " + t15.strftime("%H:%M") + " .... with ML model 1.2.0")
        predicted_Reflectivity_At_PredictionTimePlus15 = predictFunc25min(input_all[:-1])
        #actual_Reflectivity_At_PreditionTimePlus15 = loadReflectivityAtTPoz(netCDFData, timeStampPoz + 3)
        # saveData(predicted_Reflectivity_At_PredictionTimePlus15, actual_Reflectivity_At_PreditionTimePlus15, netCDFData,
        #          hour, minute, (timeStampPoz - startTimeStampPoz + 4) * acquisition_interval, "video/", None)
        out_filename_15 = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/' + radar_data_prefix\
                          + t15.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        SaveToNetCDF4(out_filename_15, predicted_Reflectivity_At_PredictionTimePlus15, netCDFData, t15)

       # prepare for the 3 next steps
        input_all = [input_all[-1], predicted_Reflectivity_At_PredictionTime,
                    predicted_Reflectivity_At_PredictionTimePlus5, predicted_Reflectivity_At_PredictionTimePlus10,
                    predicted_Reflectivity_At_PredictionTimePlus15]

if sys.argv[1] == 'latest':
    # t = time_floor(datetime.datetime.now() - datetime.timedelta(minutes = 10))
    # t = time_floor(datetime.datetime.now() , 10)
    # round the minutes to the previous 5 minutes
    # t = datetime.datetime(t.year, t.month, t.day, t.hour, int((t.minute-5)/5) * 5)
    # timestamp = t.strftime("%Y%m%dT%H%M") # e.g. 20220105T1615
    # httpLinkOrFile = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/latest/yrwms-nordic.mos.
    # pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.'+timestamp+'00Z.nc'
    # httpLinkOrFile = '/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/yrwms-nordic.mos.pcappi-0
    # -dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.'+timestamp+'00Z.nc'

    list_of_files = glob.glob('/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/*')  # storeB/project/
    httpLinkOrFile = max(list_of_files, key=os.path.getctime)
    lf = sorted(list_of_files, key=os.path.getmtime)
    # pdb.set_trace()
    # for filename in list_of_files[-3:] shutil.copy(filename, '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/.')
    timestamp = httpLinkOrFile[-19:-3]
    t = datetime.datetime(int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]), int(timestamp[9:11]),
                          int(timestamp[11:13]), int(timestamp[13:15]))
else:
    t = str(sys.argv[1])

# use 3 models
path_to_models = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/version2'
model1_0_0 = loadMLModel('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/version2/3_modele_output_5_15_25/xnow-new-ds-45days-input4steps5min-output5min.hdf5')
model1_1_0 = loadMLModel('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/version2/3_modele_output_5_15_25/xnow1.1.0-new-ds-45days-input4steps5min-output15min.hdf5')
model1_2_0 = loadMLModel('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/version2/3_modele_output_5_15_25/xnow-1.1.2-45days-input4steps5min-output25min.hdf5')

nrStepsInTheFuture = int(sys.argv[2])  # default 12
# generate multiple 1 hour in the future
generate_multiple_3models(t, nrStepsInTheFuture)

