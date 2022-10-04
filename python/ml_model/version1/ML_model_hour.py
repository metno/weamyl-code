'''
Modified: abdelkaderm@met.no 21/12/2021 (in progress)

Original script: Sample usage of a prediction model
Created on Jan 8, 2021
'''

# import requests
import os
import shutil
import glob
import sys
import datetime
import pdb
from netCDF4 import Dataset, num2date, date2num
import pyproj
import xarray as xr
# from mpl_toolkits.basemap import Basemap
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
from math import nan


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


def loadMLModel(modelFileName):
    """
      Load Trained ML Model from the file provided by the ML Development Team form a HDF5 file
      HDF5 file - contains the model's architecture, weights as resulted after the model was trained
      Model can be loaded once and used for all subsequent predictions
    """
    import keras
    return keras.models.load_model(modelFileName, compile=False)


def predictMosaic(model, input4):
    """
      inputData - 4 consecutive 2 dimensional ndarray, the raw data in the same shape and format as in the ntcdf file
      return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """
    inputForTheModel1 = prepareDataForModelMosaic(input4[0])
    inputForTheModel2 = prepareDataForModelMosaic(input4[1])
    inputForTheModel3 = prepareDataForModelMosaic(input4[2])
    inputForTheModel4 = prepareDataForModelMosaic(input4[3])
    inputForTheModel = np.stack([inputForTheModel1, inputForTheModel2, inputForTheModel3, inputForTheModel4], axis=3)

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
    # replace negatives with -1
    productRawData[(productRawData < -1)] = -1
    # replace missing data with sentinel value -5
    productRawData[np.isnan(productRawData)] = -5
    # normalize to [0,1]
    domainMin = -5
    domainMax = 75
    productRawData -= domainMin
    productRawData /= (domainMax - domainMin)

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
    # set back nan values
    prediction[(prediction < -2.5)] = nan
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


def testMosaicTiledProductSingleTimePrediction(t, httpLinkOrFile0, httpLinkOrFile1, httpLinkOrFile2, httpLinkOrFile3):
    """
     Sample code for using a MLModel: Predict for a small region, single time-step prediction
    """

    # year = '2021'; month = '08' ; day = '01' ; hour = '14' ; minute = '30'
    # response = requests.get(httpLinkOrFile +'.html')
    # print('Response status:',response.status_code)
    # if response.status_code == 200:

    aquisitionIntervall = 5;
    t_00 = t
    t_05 = (t - datetime.timedelta(minutes=5))  # .strftime("%Y%m%dT%H%M%SZ")
    t_10 = (t - datetime.timedelta(minutes=10))  # .strftime("%Y%m%dT%H%M%SZ")
    t_15 = (t - datetime.timedelta(minutes=15))  # .strftime("%Y%m%dT%H%M%SZ")

    if os.path.isfile(httpLinkOrFile0):
        # netCDFData = loadNetCDF(httpLinkOrFile)
        netCDFData0 = loadNetCDF(httpLinkOrFile0)
        netCDFData1 = loadNetCDF(httpLinkOrFile1)
        netCDFData2 = loadNetCDF(httpLinkOrFile2)
        netCDFData3 = loadNetCDF(httpLinkOrFile3)
    else:
        print('Trying to open:' + httpLinkOrFile)
        print("Error: the radar data is not available. Please try again in few minutes !")
        # sys.exit(); # ('The requested forecast is not yet available. Please try again in few minutes!')

    # e.g. load reflectivity at 14:30  1.8.2021 and the preivous 3 time steps
    x1 = loadReflectivityAt(netCDFData0, t_00.hour, t_00.minute)
    x2 = loadReflectivityAt(netCDFData1, t_05.hour, t_05.minute)
    x3 = loadReflectivityAt(netCDFData2, t_10.hour, t_10.minute)
    x4 = loadReflectivityAt(netCDFData3, t_15.hour, t_15.minute)
    # pdb.set_trace()
    # e.g. load reflectivity at 14:35  1.8.2021
    # actual_Reflectivity_At_t5 = loadReflectivityAt(netCDFData,t.hour,str(int(t.minute) + 5))
    # pdb.set_trace()
    input4 = [x1, x2, x3, x4]
    # Begin of code relevant to how we use the model for prediction
    # Load machine learning model (load once can be used for multiple predictions)
    model = loadMLModel(
        '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/ml_model/version1/xnow-new-ds-4steps-5nan.hdf5')
    # make prediction
    # End of code relevant to how we use the model for prediction
    # Predict t + 5
    Y = predictMosaic(model, input4)
    timeStamp = t + datetime.timedelta(minutes=5)
  
    outfilename = WD + 'tmp' + timeStamp.strftime("%Y%m%dT%H%M%SZ") + ".nc"
 
    # pdb.set_trace()
    print('Saving data to file:', outfilename)
    #pdb.set_trace()
    SaveToNetCDF4(outfilename, Y, netCDFData0, timeStamp)
    # predict one hour ahead
    # pdb.set_trace()

    print('Done!')
    # plot data actual vs predicted for visual inspection
    # plot1 = plt.figure(1)
    # plt.imshow(predicted_Reflectivity_At_t5)
    # plot2 = plt.figure(2)
    # plt.imshow(actual_Reflectivity_At_t5)
    # plt.show()
    # plotData(predicted_Reflectivity_At_t5,actual_Reflectivity_At_t5,netCDFData)

    # lon, lat = netCDFData.variables['lon'].data,netCDFData.variables['lat'].data
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","white","red"])
    # vmin,vmax=-50,50

    # fig, ax = plt.subplots()

    # project actual values to the map
    # predictedProjectedImage = fig.add_subplot(121)
    # predictedProjectedImage.set_title("Predicted values")
    # m = Basemap(llcrnrlat=43.5,urcrnrlat=82,llcrnrlon=-7,urcrnrlon=41)
    # X,Y = m(lon,lat)
    # draw a NASA Blue Marble image as a map background
    # m.bluemarble()
    # use a simple colormap
    # ax.pcolormesh(X,Y,predicted_Reflectivity_At_t5,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    # ax.colorbar() #, cax=cbar_ax)

    # plt.show()

# Define working directory
WD = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/'
#WD = '/home/abdelkaderm/'

#if sys.argv[1] == 'latest':
    # t = time_floor(datetime.datetime.now() - datetime.timedelta(minutes = 10))
    # t = time_floor(datetime.datetime.now() , 10)
    # round the minutes to the previous 5 minutes
    # t = datetime.datetime(t.year, t.month, t.day, t.hour, int((t.minute-5)/5) * 5)
    # timestamp = t.strftime("%Y%m%dT%H%M") # e.g. 20220105T1615
    # httpLinkOrFile = 'https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/latest/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.'+timestamp+'00Z.nc'
    # httpLinkOrFile = '/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.'+timestamp+'00Z.nc'

    # Links to files
    #httpLinkOrFile0 = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/'+ os.environ.get('USER')+'_tmp_t_00.nc'
    #httpLinkOrFile1 = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/'+ os.environ.get('USER')+'_tmp_t_05.nc'
    #httpLinkOrFile2 = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/'+ os.environ.get('USER')+'_tmp_t_10.nc'
    #httpLinkOrFile3 = '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/'+ os.environ.get('USER')+'_tmp_t_15.nc'
    # print(t)
#else:
#    t = str(sys.argv)
#    httpLinkOrFile = "https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/" + t.year + '/' + t.month + "/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000." + t.strftime(
#        "%Y%m%d") + ".nc"

# testOne2OneSingleProductSingleTimePrediction(year,month,day,hour,minute)
for i in range(0, 120, 5):
    if i == 0:
        list_of_files = glob.glob('/lustre/storeB/project/remotesensing/radar/reflectivity-nordic/latest/*')
        httpLinkOrFile = max(list_of_files, key=os.path.getctime)
        lf = sorted(list_of_files, key=os.path.getmtime)
        # pdb.set_trace()
        # for filename in list_of_files[-3:] shutil.copy(filename, '/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/.')
        timestamp = httpLinkOrFile[-19:-3]
        t = datetime.datetime(int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]), int(timestamp[9:11]),
                              int(timestamp[11:13]), int(timestamp[13:15]))
        t_00 = t
        httpLinkOrFile0 = WD + 'tmp' + t_00.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        t_05 = t - datetime.timedelta(minutes=5)
        httpLinkOrFile1 = WD + 'tmp' + t_05.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        t_10 = t - datetime.timedelta(minutes=10)
        httpLinkOrFile2 = WD + 'tmp' + t_10.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        t_15 = t - datetime.timedelta(minutes=15)
        httpLinkOrFile3 = WD + 'tmp' + t_15.strftime("%Y%m%dT%H%M%SZ") + ".nc"   
        tmp0 = httpLinkOrFile0
        tmp1 = httpLinkOrFile1
        tmp2 = httpLinkOrFile2
        tmp3 = httpLinkOrFile3
        shutil.copyfile(lf[-4], httpLinkOrFile3)        
        shutil.copyfile(lf[-3], httpLinkOrFile2)
        shutil.copyfile(lf[-2], httpLinkOrFile1)
        shutil.copyfile(lf[-1], httpLinkOrFile0)
        testMosaicTiledProductSingleTimePrediction(t, httpLinkOrFile0, httpLinkOrFile1, httpLinkOrFile2, httpLinkOrFile3)
    else:
        list_of_files = glob.glob('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/tmp*')
        #pdb.set_trace()
        lf = sorted(list_of_files, key=os.path.getmtime)    
        httpLinkOrFile0 = lf[-1]
        httpLinkOrFile1 = lf[-2]
        httpLinkOrFile2 = lf[-3]
        httpLinkOrFile3 = lf[-4]

        #shutil.copyfile(httpLinkOrFile2, httpLinkOrFile3)
        #os.remove(httpLinkOrFile2)
        #shutil.copyfile(httpLinkOrFile1, httpLinkOrFile2)
        #os.remove(httpLinkOrFile1)
        #shutil.copyfile(httpLinkOrFile0, httpLinkOrFile1)  
        timeStamp = t + datetime.timedelta(minutes=i)
        #outfilename = WD + 'tmp' + timeStamp.strftime("%Y%m%dT%H%M%SZ") + ".nc"
        #shutil.copyfile(outfilename, httpLinkOrFile0)
        testMosaicTiledProductSingleTimePrediction(timeStamp, httpLinkOrFile0, httpLinkOrFile1, httpLinkOrFile2, httpLinkOrFile3)


os.remove(tmp0)
os.remove(tmp1)
os.remove(tmp2)
os.remove(tmp3)

# for filename in list_of_files[-3:] os.remove('/lustre/storeB/project/IT/geout/weamyl/weamyl_model/tmp/'+filename)
