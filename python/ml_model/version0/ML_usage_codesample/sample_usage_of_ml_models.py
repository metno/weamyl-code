'''
Created on Jan 8, 2021

Sample usage of a prediction model

'''

import netCDF4
import pyproj
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import nan


def loadNetCDF(httpLinkOrFile="https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2021/01/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20210107.nc"):
    """
      Return the NetCDF Dataset
      Change the default value if you want to experiment with other netcdf files
      Can use links to threads but in this case data will be downloaded each times
      The URL for the current default value is: https://thredds.met.no/thredds/dodsC/remotesensing/reflectivity-nordic/2021/01/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20210107.nc
    """
    return xr.open_dataset(httpLinkOrFile)
    
def loadReflectivityAt(netCDFData,hours, minutes):
    aquisitionIntervall = 5
    timeStampPoz = (hours*60+minutes)//aquisitionIntervall
    return netCDFData.variables["equivalent_reflectivity_factor"][timeStampPoz,:,:].data


def plotData(predicted_Velocity,actual_Velocity, netCDFData):
    """
    Visualize data on a map actual vs predicted side by side
    Use Equidistant Cylindrical Projection 
    Show approximately the same area as in Godiva viewer from thredds.met.no 
    """            
    lon, lat = netCDFData.variables['lon'].data,netCDFData.variables['lat'].data
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","white","red"])
    vmin,vmax=-50,50
    
    fig = plt.figure()
    #project actual values to the map
    actualProjectedImage = fig.add_subplot(121)
    actualProjectedImage.set_title("Actual values")
    m = Basemap(llcrnrlat=43.5,urcrnrlat=82,llcrnrlon=-7,urcrnrlon=41)    
    X,Y = m(lon,lat)    
    #draw a NASA Blue Marble image as a map background
    m.bluemarble()
    #use a simple colormap
    actualProjectedImage.pcolormesh(X,Y,actual_Velocity,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    #project predicted values to the map
    predictedProjectedImage = fig.add_subplot(122)
    predictedProjectedImage.set_title("Predicted values")
    m = Basemap(llcrnrlat=43.5,urcrnrlat=82,llcrnrlon=-7,urcrnrlon=41)
    X,Y = m(lon,lat)
    #draw a NASA Blue Marble image as a map background
    m.bluemarble()    
    im = predictedProjectedImage.pcolormesh(X,Y,predicted_Velocity,cmap=cmap,vmin=vmin,vmax=vmax,shading='auto')
    
    #show colormap
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()
    
def loadMLModel(modelFileName):
    """
      Load Trained ML Model from the file provided by the ML Development Team form a HDF5 file
      HDF5 file - contains the model's architecture, weights as resulted after the model was trained
      Model can be loaded once and used for all subsequent predictions      
    """
    import keras    
    return keras.models.load_model(modelFileName, compile = False)    


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
    return productRawData[None,... ]

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
    prediction[prediction>89] = nan
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
    #replace missing data / negative data
    productRawData[np.isnan(productRawData)] = 300
    
    #break into tiles
    slices = sliding_window(productRawData, 128)
    slices = slices.reshape(-1,*slices.shape[2:])    
    return slices

def sliding_window(a, window):
    """Very basic multi dimensional rolling window. window should be the shape of
    of the desired subarrays. Window is either a scalar or a tuple of same size
    as `arr.shape`.
    """    
    h,w = a.shape
    bh,bw = window,window
    shape = (h//bh, w//bw, bh, bw)    
    strides = a.itemsize*np.array([w*bh,bw,w,1])
    blocks=np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
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
    #special treatment for some values
    prediction[prediction>89] = nan
    #stitch tiles together to form a single output matrix    
    return combineToASingleMAtrix(prediction)

def combineToASingleMAtrix(prediction):
    """
    given a list of 2d matrix (mosaic tiles)
    reconstruct the larger matrix
    return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data
    """    
    tiles = prediction.reshape(16, 13,128,128)
    tiles = tiles.swapaxes(1, 2)
    tiles = tiles.reshape(2048, 1664)
    #pad with nan to the exact dimension 
    result = np.empty((2134, 1694))
    result[:] = nan
    result[:tiles.shape[0],:tiles.shape[1]] = tiles
    return result
    
    
def testMosaicTiledProductSingleTimePrediction():
    """
     Sample code for using a MLModel: Predict for a small region, single time-step prediction
    """
    netCDFData = loadNetCDF()    
    #load reflectivity at 14:30  1.8.2021
    actual_Reflectivity_At_14_30 = loadReflectivityAt(netCDFData,14,30)
    #load reflectivity at 14:35  1.8.2021
    actual_Reflectivity_At_14_35 = loadReflectivityAt(netCDFData,14,35)
    
    #Begin of code relevant to how we use the model for prediction
    #Load machine learning model (load once can be used for multiple predictions)
    model = loadMLModel('128x128_trained_135ep_20210107.hdf5')
    
    #make prediction
    predicted_Reflectivity_At_14_35 = predictMosaic(model, actual_Reflectivity_At_14_30)    
    #End of code relevant to how we use the model for prediction
    
    #plot data actual vs predicted for visual inspection
    plotData(predicted_Reflectivity_At_14_35,actual_Reflectivity_At_14_35,netCDFData)
    
    
def testOne2OneSingleProductSingleTimePrediction():
    """
     Sample code for using a MLModel: One to one approach single time-step prediction
    """
    netCDFData = loadNetCDF()    
    #load reflectivity at 14:30  1.8.2021
    actual_Reflectivity_At_14_30 = loadReflectivityAt(netCDFData,14,30)
    #load reflectivity at 14:35  1.8.2021
    actual_Reflectivity_At_14_35 = loadReflectivityAt(netCDFData,14,35)
    
    #Begin of code relevant to how we use the model for prediction
    #Load machine learning model (load once can be used for multiple predictions)
    model = loadMLModel('700ep_trained_with_20210107.hdf5')
    
    #make prediction
    predicted_Reflectivity_At_14_35 = predictOne2One(model, actual_Reflectivity_At_14_30)    
    #End of code relevant to how we use the model for prediction
    
    # save predicted reflectivity data to NetCDF file
    print(predicted_Reflectivity_At_14_35)
    #plot data actual vs predicted for visual inspection
    plotData(predicted_Reflectivity_At_14_35,actual_Reflectivity_At_14_35,netCDFData)

# testOne2OneSingleProductSingleTimePrediction()
testMosaicTiledProductSingleTimePrediction()
