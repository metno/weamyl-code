
import netCDF4
import pyproj
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from copy import deepcopy


# httpLinkOrFile=
# httpLinkOrFile="data/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20210107.nc"
# httpLinkOrFile="data/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20220316.nc"
def loadNetCDF(httpLinkOrFile="data/yrwms-nordic.mos.pcappi-0-dbz.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.20220316.nc"):
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
    #model.summary()
    return model   

def loadReflectivityAtTPoz(netCDFData, timeStampPoz):
    return netCDFData.variables["equivalent_reflectivity_factor"][timeStampPoz, :, :].data

def saveData(predicted_Velocity, actual_Velocity, netCDFData, hour, minute,predictionMinutes,dir,secondPredicted_Velocity=None):
    fig = createFigure(predicted_Velocity, actual_Velocity, netCDFData, hour, minute,predictionMinutes,False,secondPredicted_Velocity);    
    plt.savefig(f"{dir}testsave{hour:02d}{minute:02d}.png", bbox_inches='tight',dpi=400)  
    plt.close(fig);  
    
def createFigure(predicted_Velocity, actual_Velocity, netCDFData, hour, minute,predictionMinutes,showColorMap=True,secondPredicted_Velocity=None):
    """
    Visualize data on a map actual vs predicted side by side
    Use Equidistant Cylindrical Projection 
    Show approximately the same area as in Godiva viewer from thredds.met.no 
    """    
    lon, lat = netCDFData.variables['lon'].data, netCDFData.variables['lat'].data
    cmap, vmin, vmax = createColormap()    
    fig = plt.figure()
    llcrnrlat=52 #43.5 #Y down
    urcrnrlat=75 #82  #Y up
    llcrnrlon=-2 #-7  #X left
    urcrnrlon=35  #41  #X right
    
    nrPlots = 2
    if not secondPredicted_Velocity is None: nrPlots=3
    
    # project actual values to the map
    subplotActual = fig.add_subplot(1,nrPlots,1)
    drawPrediction(subplotActual,actual_Velocity,lon,lat,llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap, vmin, vmax,f"Radar at {hour:02d}:{minute:02d}")    
    
    # project predicted values to the map
    subplotPredict = fig.add_subplot(1,nrPlots,2)
    im = drawPrediction(subplotPredict,predicted_Velocity,lon,lat,llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap, vmin, vmax,f"Prediction ({predictionMinutes} min)")
    
    if not secondPredicted_Velocity is None:
        subplotPredict = fig.add_subplot(1,nrPlots,3)
        im = drawPrediction(subplotPredict,secondPredicted_Velocity,lon,lat,llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap, vmin, vmax,"Prediction2")

    if showColorMap:
        # show colormap
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
        fig.colorbar(im, cax=cbar_ax)
    return fig

def drawPrediction(subplot,predicted_Velocity,lon,lat,llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon, cmap, vmin, vmax,title):        
    subplot.set_title(title)
    m = Basemap(llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon)
    X, Y = m(lon, lat)
    # draw a NASA Blue Marble image as a map background
    m.bluemarble()
    im = subplot.pcolormesh(X, Y, predicted_Velocity, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')    
    return im

def createColormapMet():
    #R,G,B   Reflectivity
    #0.925,0.882,0.878   >53.5
    #0.824,0.576,0.796   39
    #0.788,0.220,0.149   34
    #0.914,0.482,0.149   29.5
    #0.914,0.592,0.149   23
    #0.914,0.698,0.149   18
    #0.914,0.808,0.149   14.5
    #0.914,0.914,0.149   7
    #0.753,0.753,0.329   2
    #0.573,0.573,0.486  -2.5    
    colors = [(0.0,(0.573, 0.573, 0.486)), (0.08035,(0.753, 0.753, 0.329)), (0.16964,(0.914, 0.914, 0.149)), 
              (0.30357,(0.914, 0.808, 0.149)), (0.36607,(0.914, 0.698, 0.149)), (0.45535,(0.914, 0.592, 0.149)), 
              (0.57142,(0.914, 0.482, 0.149)), (0.65178,(0.788, 0.22, 0.149)), (0.74107,(0.824, 0.576, 0.796)), 
              (1.0,(0.925, 0.882, 0.878))]
    cmap_name = 'Reflectivity'
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))
    vmin, vmax = -2.5, 53.5
    return cmap,vmin,vmax
    
def createColormap():
    return createColormapMet()

def predictXNOW_NAN_2_Negative_1_4TInp(model, inputDataAll,replaceInputNanWith,outputNanTreashold):
    """
      inputData - 2 dimensional ndarray, the raw data in the same shape and format as in the ntcdf file
      return 2 dimensional ndarray, predicted raw product values. The shape, format, coordinate system, etc is the same as in the source netcdf data 
    """
    inputForTheModel1 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[0],replaceInputNanWith)
    inputForTheModel2 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[1],replaceInputNanWith)
    inputForTheModel3 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[2],replaceInputNanWith)
    inputForTheModel4 = prepareDataForModelXNOW_NAN_2_Negative_1(inputDataAll[3],replaceInputNanWith)    
    inputForTheModel = np.stack([inputForTheModel1,inputForTheModel2,inputForTheModel3,inputForTheModel4],axis=3)
    
    prediction = model.predict(inputForTheModel)
    # prediction = inputForTheModel
    return postProcessPredictionXNOW_NAN_2_Negative_1(prediction,outputNanTreashold)

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
    #put nan if prediction is -2 or less  
    # print("replace output <",outputNanTreashold," with nan")      
    prediction[(prediction < outputNanTreashold) ] = nan
    
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
    
def prepareDataForModelXNOW_NAN_2_Negative_1(productRawData,replaceInputNanWith):
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
    
    #replace negatives with -1
    productRawData[(productRawData < -1)] = -1

    #replace nan with -2
    # print("replace input nan with ",replaceInputNanWith)
    productRawData[np.isnan(productRawData)] = replaceInputNanWith
    
    #normalize to [0,1]
    domainMin=replaceInputNanWith
    domainMax=75
    productRawData-=domainMin
    productRawData /= (domainMax-domainMin)
    
    # break into tiles
    slices = sliding_window(productRawData, 128)
    slices = slices.reshape(-1,*slices.shape[2:], 1)
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

def generateMultiple3Models(predictFunc5min,predictFunc15min,predictFunc25min,nrStepsInTheFuture):
    netCDFData = loadNetCDF()
    for startp in range(5,netCDFData.variables["equivalent_reflectivity_factor"].shape[0]-nrStepsInTheFuture,nrStepsInTheFuture+5):
        generate3Models(predictFunc5min,predictFunc15min,predictFunc25min,startp,nrStepsInTheFuture)
    
def generate3Models(predictFunc5min,predictFunc15min,predictFunc25min,startTimeStampPoz,nrStepsInTheFuture,netCDFData = None):
    """
     Sample code for using a MLModel: One to one approach single time-step prediction
    """
    if netCDFData==None:
        netCDFData = loadNetCDF()
    aquisitionIntervall = 5    
    # load reflectivity at 14:30  1.8.2021    
    actual_Reflectivity_At_Minus5 = loadReflectivityAtTPoz(netCDFData, startTimeStampPoz-5)
    actual_Reflectivity_At_Minus4 = loadReflectivityAtTPoz(netCDFData, startTimeStampPoz-4)
    actual_Reflectivity_At_Minus3 = loadReflectivityAtTPoz(netCDFData, startTimeStampPoz-3)
    actual_Reflectivity_At_Minus2 = loadReflectivityAtTPoz(netCDFData, startTimeStampPoz-2)
    actual_Reflectivity_At_Minus1 = loadReflectivityAtTPoz(netCDFData, startTimeStampPoz-1)
    inputAll = [actual_Reflectivity_At_Minus5,actual_Reflectivity_At_Minus4,actual_Reflectivity_At_Minus3,actual_Reflectivity_At_Minus2,actual_Reflectivity_At_Minus1]
        
    # make prediction (feed result back in until we have the desired timestep in the future
    # pdb.set_trace()    
    # import pdb; pdb.set_trace()
    for timeStampPoz in range(startTimeStampPoz,startTimeStampPoz+nrStepsInTheFuture,4):
        hour = timeStampPoz * aquisitionIntervall // 60
        minute = timeStampPoz * aquisitionIntervall % 60        
        print(f"Generate for {hour:02d}:{minute:02d}.... with 1.0.0")
        predicted_Reflectivity_At_PredictionTime = predictFunc5min(inputAll[1:])
        actual_Reflectivity_At_PreditionTime = loadReflectivityAtTPoz(netCDFData, timeStampPoz)
        saveData(predicted_Reflectivity_At_PredictionTime, actual_Reflectivity_At_PreditionTime, netCDFData,hour,minute,(timeStampPoz-startTimeStampPoz+1)*aquisitionIntervall,"video/",None)
        
        
        hour = (timeStampPoz+1) * aquisitionIntervall // 60
        minute = (timeStampPoz+1) * aquisitionIntervall % 60
        print(f"Generate for {hour:02d}:{minute:02d}.... with 1.1.0")
        
        predicted_Reflectivity_At_PredictionTimePlus5 = predictFunc15min(inputAll[:-1])
        actual_Reflectivity_At_PreditionTimePlus5 = loadReflectivityAtTPoz(netCDFData, timeStampPoz+1)
        saveData(predicted_Reflectivity_At_PredictionTimePlus5, actual_Reflectivity_At_PreditionTimePlus5, netCDFData,hour,minute,(timeStampPoz-startTimeStampPoz+2)*aquisitionIntervall,"video/",None)
                
        hour = (timeStampPoz+2) * aquisitionIntervall // 60
        minute = (timeStampPoz+2) * aquisitionIntervall % 60   
        print(f"Generate for {hour:02d}:{minute:02d}.... with 1.1.0")
        predicted_Reflectivity_At_PredictionTimePlus10 = predictFunc15min(inputAll[1:])
        actual_Reflectivity_At_PreditionTimePlus10 = loadReflectivityAtTPoz(netCDFData, timeStampPoz+2)
        saveData(predicted_Reflectivity_At_PredictionTimePlus10, actual_Reflectivity_At_PreditionTimePlus10, netCDFData,hour,minute,(timeStampPoz-startTimeStampPoz+3)*aquisitionIntervall,"video/",None)
        
        hour = (timeStampPoz+3) * aquisitionIntervall // 60
        minute = (timeStampPoz+3) * aquisitionIntervall % 60   
        print(f"Generate for {hour:02d}:{minute:02d}.... with 1.1.1")        
        predicted_Reflectivity_At_PredictionTimePlus15 = predictFunc25min(inputAll[:-1])
        actual_Reflectivity_At_PreditionTimePlus15 = loadReflectivityAtTPoz(netCDFData, timeStampPoz+3)
        saveData(predicted_Reflectivity_At_PredictionTimePlus15, actual_Reflectivity_At_PreditionTimePlus15, netCDFData,hour,minute,(timeStampPoz-startTimeStampPoz+4)*aquisitionIntervall,"video/",None)
        
                
        #prepare for the 3 next step
        inputAll=[inputAll[-1],predicted_Reflectivity_At_PredictionTime,predicted_Reflectivity_At_PredictionTimePlus5,predicted_Reflectivity_At_PredictionTimePlus10,predicted_Reflectivity_At_PredictionTimePlus15]

#use 3 models
model1_0_0 = loadMLModel('xnow1.1.2-output5min.hdf5')
model1_1_0 = loadMLModel('xnow1.1.0-output15min.hdf5')
model1_2_0 = loadMLModel('xnow-1.1.1-output25min.hdf5')
#generate multiple 1 hour in the future
generateMultiple3Models(lambda input:predictXNOW_NAN_2_Negative_1_4TInp(model1_0_0,input,-5,-2.5),lambda input:predictXNOW_NAN_2_Negative_1_4TInp(model1_1_0,input,-5,-2.5),lambda input:predictXNOW_NAN_2_Negative_1_4TInp(model1_2_0,input,-5,-2.1),12)
