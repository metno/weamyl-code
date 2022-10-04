# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

'''
Modified: abdelkaderm@met.no 21/12/2021 (in progress)

Original script: Sample usage of a prediction model
Created on Jan 8, 2021
'''

import os
import glob
import sys
import datetime
import pdb
from netCDF4 import Dataset, num2date, date2num
import pyproj
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from dateutil.tz import tzutc

def loadNetCDF(httpLinkOrFile):
    """
      Return the NetCDF Dataset
      Change the default value if you want to experiment with other netcdf files
      Can use links to threads but in this case data will be downloaded each times
      The URL for the current default value is: https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2021/01/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20210107.nc
    """
    # Example year = '2021'; month = '01' ; day = '07'

    return xr.open_dataset(httpLinkOrFile)


def loadReflectivityAt(netCDFData, hour, minute):
    aquisitionIntervall = 5
    if sys.argv[1] == 'latest':
        timeStampPoz = 0
    else:
        timeStampPoz = (hour * 60 + minute) // aquisitionIntervall

    return netCDFData.variables["equivalent_reflectivity_factor"][timeStampPoz, :, :].data


def plotData(predicted, actual, netCDFData):
    """
    Visualize data on a map actual vs predicted side by side
    Use Equidistant Cylindrical Projection
    Show approximately the same area as in Godiva viewer from thredds.met.no
    """
    lon, lat = netCDFData.variables['lon'].data, netCDFData.variables['lat'].data
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    vmin, vmax = -50, 50

    fig = plt.figure()
    # project actual values to the map
    actualProjectedImage = fig.add_subplot(121)
    actualProjectedImage.set_title("Actual values")
    m = Basemap(llcrnrlat=43.5, urcrnrlat=82, llcrnrlon=-7, urcrnrlon=41)
    X, Y = m(lon, lat)
    # draw a NASA Blue Marble image as a map background
    m.bluemarble()
    # use a simple colormap
    actualProjectedImage.pcolormesh(X, Y, actual, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # project predicted values to the map
    predictedProjectedImage = fig.add_subplot(122)
    predictedProjectedImage.set_title("Predicted values")
    m = Basemap(llcrnrlat=43.5, urcrnrlat=82, llcrnrlon=-7, urcrnrlon=41)
    X, Y = m(lon, lat)
    # draw a NASA Blue Marble image as a map background
    m.bluemarble()
    im = predictedProjectedImage.pcolormesh(X, Y, predicted, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # show colormap
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()


def SaveToNetCDF4(outfilename, predicted, netCDFData, timeStamp):
    """
      Save the predicted data into a similar NetCDF file by replacing original values with predicted values
    """

    ncfile = Dataset(outfilename, mode='w', format='NETCDF4_CLASSIC')
    ncfile.title = 'Machine learning model version 0 output for WeaMyL project'
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
    Xcs = ncfile.createVariable('X', np.float32, ('X',))
    Xcs[:] = netCDFData.variables['Xc'].data
    Xcs.axis = netCDFData.variables['Xc'].attrs['axis']
    Xcs.standard_name = netCDFData.variables['Xc'].attrs['standard_name']
    Xcs.units = netCDFData.variables['Xc'].attrs['units']

    Ycs = ncfile.createVariable('Y', np.float32, ('Y',))
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
    #projs.lambert_conformal_conic: false_easting = 0.;
    #projs.lambert_conformal_conic: false_northing = 0.;

    times = ncfile.createVariable('time', np.int32, ('time',))
    times.axis = netCDFData.variables['time'].attrs['axis']
    times.long_name = 'time'
    times.standard_name = netCDFData.variables['time'].attrs['standard_name']
    times.units = "seconds since 1970-01-01 00:00:00 +00:00"
    times.calendar = "proleptic_gregorian"
    # times[:]= date2num(np.datetime64(netCDFData.variables['time'].data[0],'s').astype(datetime.datetime),  times.units)
    times[:] = date2num(timeStamp, times.units, calendar = times.calendar)
    print("time values (in units {}):\n{}".format(times.units, times[:]))
    # pdb.set_trace()
    rf = ncfile.createVariable('equivalent_reflectivity_factor', np.float32, ('time', 'Y', 'X'), fill_value=9.9621e+36)
    # predicted_Reflectivity[np.isnan(predicted_Reflectivity)] = 0
    rf[0, :, :] = predicted

    # np.random.random((2134,1694)) #predicted_Reflectivity
    rf.units = netCDFData.variables['equivalent_reflectivity_factor'].attrs['units']
    rf.long_name = netCDFData.variables['equivalent_reflectivity_factor'].attrs['long_name']
    rf.standard_name = netCDFData.variables['equivalent_reflectivity_factor'].attrs['standard_name']
    rf.coordinates = 'lon lat'
    rf.grid_mapping = netCDFData.variables['equivalent_reflectivity_factor'].attrs['grid_mapping']

    ncfile.close()


def loadMLModel(modelFileName):
    """
      Load Trained ML Model from the file provided by the ML Development Team form a HDF5 file
      HDF5 file - contains the model's architecture, weights as resulted after the model was trained
      Model can be loaded once and used for all subsequent predictions
    """
    import keras
    return keras.models.load_model(modelFileName, compile=False)


def predictOne2One(model, inputData):
    """
      inputData - 2 dimensional ndarray, the raw data in the same shape and format as in the ntcdf file
      return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    inputForTheModel = prepareDataForModelOne2One(inputData)
    prediction = model.predict(inputForTheModel)
    return postProcessPredictionOne2One(prediction)


def prepareDataForModelOne2One(productRawData):
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
    productRawData[np.isnan(productRawData)] = 300
    return productRawData[None, ...]


def postProcessPredictionOne2One(prediction):
    """
      Some post-processing of the predicted data may be necessary (crop/stitch/normalize  etc)
      For every model we will provide the description of the necessary steps and transformations
      Sample processing:
          -  stitch data predicted for smaller regions into a large region
          -  mark some values as missing (nan)
          -  divide results into multiple results (ex: predictOne2One multiple values for different timestamps in a single step)
      prediction - 2 dimensional ndarray, data as resulted form the prediction
      return 2 dimensional ndarray,predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    prediction = prediction[0]
    prediction[prediction > 89] = nan
    return prediction


def predictMosaic(model, inputData):
    """
      inputData - 2 dimensional ndarray, the raw data in the same shape and format as in the ntcdf file
      return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    inputForTheModel = prepareDataForModelMosaic(inputData)
    prediction = model.predict(inputForTheModel)
    return postProcessPredictionMosaic(prediction)


def prepareDataForModelMosaic(productRawData):
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
    # replace missing data / negative data
    productRawData[np.isnan(productRawData)] = 300

    # break into tiles
    slices = sliding_window(productRawData, 128)
    slices = slices.reshape(-1, *slices.shape[2:])
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


def postProcessPredictionMosaic(prediction):
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
    # special treatment for some values
    prediction[prediction > 89] = nan
    # stitch tiles together to form a single output matrix
    return combineToASingleMAtrix(prediction)


def combineToASingleMAtrix(prediction):
    """
    given a list of 2d matrix (mosaic tiles)
    reconstruct the larger matrix
    return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    tiles = prediction.reshape(16, 13, 128, 128)
    tiles = tiles.swapaxes(1, 2)
    tiles = tiles.reshape(2048, 1664)
    # pad with nan to the exact dimension
    result = np.empty((2134, 1694))
    result[:] = nan
    result[:tiles.shape[0], :tiles.shape[1]] = tiles
    return result


def testMosaicTiledProductSingleTimePrediction(t, httpLinkOrFile):
    """
     Sample code for using a MLModel: Predict for a small region, single time-step prediction
    """

    # year = '2021'; month = '08' ; day = '01' ; hour = '14' ; minute = '30'
    # response = requests.get(httpLinkOrFile +'.html')
    # print('Response status:',response.status_code)
    # if response.status_code == 200:
    if os.path.isfile(httpLinkOrFile):
        netCDFData = loadNetCDF(httpLinkOrFile)
    else:
        print('Trying to open:' + httpLinkOrFile)
        print("Error: the radar data is not available. Please try again in few minutes !")
        # sys.exit(); # ('The requested forecast is not yet available. Please try again in few minutes!')

    aquisitionIntervall = 5;
    # e.g. load reflectivity at 14:30  1.8.2021
    actual_Reflectivity_At_t0 = loadReflectivityAt(netCDFData, t.hour, t.minute)
    # e.g. load reflectivity at 14:35  1.8.2021
    # actual_Reflectivity_At_t5 = loadReflectivityAt(netCDFData,t.hour,str(int(t.minute) + 5))

    # Begin of code relevant to how we use the model for prediction
    # Load machine learning model (load once can be used for multiple predictions)
    model = loadMLModel('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/version0/ML_usage_codesample/128x128_trained_135ep_20210107.hdf5')

    # make prediction
    # End of code relevant to how we use the model for prediction
    # predict step t5 = t0 + 5 from t0
    X5 = predictMosaic(model, actual_Reflectivity_At_t0)
    # Loop through following time steps e.g. t + 10, t + 15, etc. until t+55
    for i in range(10, 60, 5):
        Y = predictMosaic(model, X5)
        timeStamp = t + datetime.timedelta(minutes=i)
        outfilename = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/output/version0/'+ os.environ.get('USER') +'_tmp' + timeStamp.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        # pdb.set_trace()
        print('Saving data to file:', outfilename)
        SaveToNetCDF4(outfilename, Y, netCDFData, timeStamp)
        del X5
        X5 = Y
        del Y

    print('Done!')
    # plot data actual vs predicted for visual inspection
    # plot1 = plt.figure(1)
    # plt.imshow(predicted_Reflectivity_At_t5)
    # plot2 = plt.figure(2)
    # plt.imshow(actual_Reflectivity_At_t5)
    # plt.show()
    # plotData(predicted_Reflectivity_At_t5,actual_Reflectivity_At_t5,netCDFData)

    #lon, lat = netCDFData.variables['lon'].data, netCDFData.variables['lat'].data
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    #vmin, vmax = -50, 50

    #fig, ax = plt.subplots()

    # project actual values to the map
    # predictedProjectedImage = fig.add_subplot(121)
    # predictedProjectedImage.set_title("Predicted values")
    #m = Basemap(llcrnrlat=43.5, urcrnrlat=82, llcrnrlon=-7, urcrnrlon=41)
    #X, Y = m(lon, lat)
    # draw a NASA Blue Marble image as a map background
    #m.bluemarble()
    # use a simple colormap
    #ax.pcolormesh(X, Y, predicted_Reflectivity_At_t5, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    # ax.colorbar() #, cax=cbar_ax)

    #plt.show()


def testOne2OneSingleProductSingleTimePrediction(year, month, day, hour, minute):
    """
     Sample code for using a MLModel: One to one approach single time-step prediction
    """

    year = '2021';
    month = '08';
    day = '01';
    hour = '14';
    minute = '30'

    netCDFData = loadNetCDF(year, month, day)

    # load reflectivity at 14:30  1.8.2021
    actual_Reflectivity_At_t0 = loadReflectivityAt(netCDFData, hour, minute)
    # load reflectivity at 14:35  1.8.2021
    actual_Reflectivity_At_t5 = loadReflectivityAt(netCDFData, hour, str(int(minute) + 5))

    # Begin of code relevant to how we use the model for prediction
    # Load machine learning model (load once can be used for multiple predictions)
    model = loadMLModel('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/ML_usage_codesample/128x128_trained_135ep_20210107.hdf5')

    # make prediction
    predicted_Reflectivity_At_t5 = predictOne2One(model, actual_Reflectivity_At_t0)
    # End of code relevant to how we use the model for prediction

    # save predicted reflectivity data to NetCDF file
    # predicted_Reflectivity_At_14_35.to_netcdf(path = './output/test1.nc')

    outfilename = os.environ.get('USER') + "_predicted_reflectivity_yrwms-nordic_mos_pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block_nordiclcc-1000_" + year + month + day + "T" + hour + str(
        (int(minute) + 5)) + ".nc"

    print('Saving data to file:', outfilename)
    timeStampPoz2 = (int(hour) * 60 + int(minute)) // aquisitionIntervall + 1
    SaveToNetCDF4(outfilename, predicted_Reflectivity_At_t5, netCDFData, timeStampPoz2)
    print('Done!')
    # plot data actual vs predicted for visual inspection
    # plotData(predicted_Reflectivity_At_t5,actual_Reflectivity_At_t5,netCDFData)
    plot1 = plt.figure(1)
    plt.imshow(predicted_Reflectivity_At_t5)
    plot2 = plt.figure(2)
    plt.imshow(actual_Reflectivity_At_t5)
    plt.show()


if sys.argv[1] == 'latest':
    # t = time_floor(datetime.datetime.now() - datetime.timedelta(minutes = 10))
    # t = time_floor(datetime.datetime.now() , 10)
    # round the minutes to the previous 5 minutes
    # t = datetime.datetime(t.year, t.month, t.day, t.hour, int((t.minute-5)/5) * 5)
    # timestamp = t.strftime("%Y%m%dT%H%M") # e.g. 20220105T1615
    # httpLinkOrFile = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/latest/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.'+timestamp+'00Z.nc'
    # httpLinkOrFile = '/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.'+timestamp+'00Z.nc'
    list_of_files = glob.glob('/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/*')
    httpLinkOrFile = max(list_of_files, key=os.path.getctime)
    timestamp = httpLinkOrFile[-19:-3]
    t = datetime.datetime(int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]), int(timestamp[9:11]), int(timestamp[11:13]), int(timestamp[13:15]), 000000, tzinfo=tzutc())
else:
    t = str(sys.argv)
    httpLinkOrFile = "https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/" + str(t.year) + '/' + str(t.month) + "/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000." + t.strftime(
        "%Y%m%d") + ".nc"

# testOne2OneSingleProductSingleTimePrediction(year,month,day,hour,minute)
testMosaicTiledProductSingleTimePrediction(t, httpLinkOrFile)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

