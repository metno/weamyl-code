#!/usr/bin/env python

# Copyright 2021 Trygve Aspenes <trygveas@met.no>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The intention with this script is to make well defined products from METEOSAT SEVIRI
data for the WeaMyL project and save the data to netcdf.

To read the data pytroll/satpy is used. You will find satpy package on conda-forge or pypi
or you can clone or fork from https://github.com/pytroll/satpy.git

METEOSAT SEVIRI data is collected every 15 minutes placed over equator at the 0 deg longitude.
The SEVIRI sensor has 12 channels; 11 in the visible, nearinfrared and infrared including two 
water vapour channel in 3km resolution in an 3712x3712 pixel. And one visible channel(HRV) in 1km 
resolution with 5568x11136 covering half the disk in two different section.

Extra care needs to be taken to match the 1km with the 3km channels.
"""

import glob
import numpy as np
from satpy import Scene
from netCDF4 import Dataset
from satpy.utils import debug_on
from datetime import datetime, timezone
from satpy.writers import get_enhanced_image
from acdd_and_additional_elements import pl_sn, pl_ln, instr_sn, instr_ln, pr, ir
debug_on()

netcdf_global_attr = {}
netcdf_global_attr['collection'] = 'WeaMyL'

netcdf_global_attr['title'] = 'Meteosat data formated for the WeaMyL project'
netcdf_global_attr['title_lang'] = 'en'
# MMD abstract, ACDD summary
netcdf_global_attr['summary'] = 'Meteosat data L1B data formated to various products to identify specific '
netcdf_global_attr['summary'] += 'meteorological phenonmen.'
netcdf_global_attr['summary_lang'] = 'en'
# MMD temporal_extent,start_date, ACDD time_coverage_start

netcdf_global_attr['processing_level'] = 'Operational'
# MMD use_constraint_identifier, ACDD license
netcdf_global_attr['license'] = 'Copyright 2021, EUMETSAT, All Rights Reserved'
netcdf_global_attr['license_resource'] = 'https://www-cdn.eumetsat.int/files/2021-01/45173%20Data_Policy.pdf'

# MMD personnel, ACDD creator
netcdf_global_attr['creator_type'] = 'institution'
netcdf_global_attr['creator_role'] = 'Technical contact'
netcdf_global_attr['creator_name'] = 'DIVISION FOR OBSERVATION QUALITY AND DATA PROCESSING'
netcdf_global_attr['creator_email'] = 'post@met.no'
netcdf_global_attr['creator_url'] = 'met.no'
netcdf_global_attr['creator_institution'] = 'MET NORWAY'

netcdf_global_attr['contributor_role'] = 'Metadata author'
netcdf_global_attr['contributor_name'] = 'DIVISION FOR OBSERVATION QUALITY AND DATA PROCESSING'
netcdf_global_attr['contributor_email'] = 'post@met.no'

netcdf_global_attr['publisher_institution'] = 'MET NORWAY'
netcdf_global_attr['publisher_country'] = 'NORWAY'
netcdf_global_attr['publisher_email'] = 'post@met.no'
netcdf_global_attr['publisher_name'] = 'DIVISION FOR OBSERVATION QUALITY AND DATA PROCESSING'
netcdf_global_attr['publisher_url'] = 'met.no'
netcdf_global_attr['keywords_vocabulary'] = 'GCMD'
netcdf_global_attr['keywords'] = 'Earth Science>Atmosphere>Atmospheric radiation'

netcdf_global_attr['institution'] = 'MET NORWAY'
netcdf_global_attr['project'] = 'Govermental core service'

# MMD mandatory not in ACDD
netcdf_global_attr['metadata_status'] = 'Active'
netcdf_global_attr['dataset_production_status'] = 'Complete'
netcdf_global_attr['iso_topic_category'] = 'climatologyMeteorologyAtmosphere,environment,oceans'

netcdf_global_attr['source'] = 'Space Borne Instrument'

files = glob.glob("/data/pytroll/testdata/seviri/20200828/H-000-MSG4__-MSG4*202008271200-__")
global_scene = Scene(reader="seviri_l1b_hrit", filenames=files)
print(global_scene.available_dataset_names())
print(global_scene.available_composite_names())

# Products agreed to include in this dataset
# The Day Microphysics RGB (VIS 0.8, MIR, 3.9, IR 10.8)
# The 24 hour Microphysics RGB (IR12-IR10.8, IR10.8-3.9/8.7, IR10.8)
# The Severe Storm RGB (WV6.2-WV7.3, IR3.9-IR10.8, NIR1.6-VIS0.6)
# The HRV Severe Storm (HRV, HRV, IR10.8-IR3.9)
# The Airmass RGB (WV6.2-WV7.3, IR9.7-IR10.8, WV6.2)
# The High Resolution VIS RGB (HRV, HRV, 10.8)
# WV6.2

# Specify satpy products
variables = ['day_microphysics', 'night_microphysics', 'microphysics_24_7',
             'convection', 'hrv_severe_storms', 'airmass', 'hrv_clouds', 'water_vapour_62']
# Load the products
global_scene.load(variables, upper_right_corner='NE')

# Resample using coarsest area (is the 3km channel product) using the native resampler
# This is needed to resample the HRC 1km channel to match the 3km channels
local_scene = global_scene.resample(global_scene.coarsest_area(), resampler='native')

# Get the longitue and latitude calues from the data
longitude, latitude = local_scene[variables[0]].attrs['area'].get_lonlats()

# Get the projection coordinates
pxc = local_scene[variables[0]].attrs['area'].projection_x_coords
pyc = local_scene[variables[0]].attrs['area'].projection_y_coords

# Make the enhanced composites as defined by satpy
data_enh = {}
for product in variables:
    data = get_enhanced_image(local_scene[product])
    data_enh[product] = data.data.compute()

# Define the area included in the dataset.
# I use linenumbers in the disk image
# Using the center 1/2 of the disk in x dir. (east-west)
start_x = 928 - 1
end_x = 2784 - 1
width = end_x - start_x
# Using the top 1/4 of the disk i y dir (north-south)
start_y = 52
end_y = 928 + 52  # The top 52 pixels are just space pixels
height = end_y - start_y

acq_time = np.flip(global_scene['IR_108'].coords['acq_time'][3712 - end_y:3712 - start_y].data)
unix_epoch = np.datetime64(0, 's')
one_second = np.timedelta64(1, 's')
# Write data to netcdf
# filename = "WeaMyL-Meteosat-FES-Europe-{:%Y%m%d%H%M%S}.nc".format(local_scene.attrs['start_time'])
filename = "WeaMyL-Meteosat-FES-Europe-{:%Y%m%d%H%M%S}.nc".format(datetime.utcfromtimestamp((acq_time[-1] - unix_epoch) / one_second))

rootgrp = Dataset(filename, "w", format="NETCDF4")

# Create dimensions
rgb = rootgrp.createDimension("rgb", 3)
x = rootgrp.createDimension("nx", width)
y = rootgrp.createDimension("ny", height)
time = rootgrp.createDimension("time")

# Set global attributes from netcdf_global_attr
for attrib in netcdf_global_attr:
    setattr(rootgrp, attrib, netcdf_global_attr[attrib])

# Create variables
rgbs = rootgrp.createVariable("rgb", "i1", ("rgb",))

xs = rootgrp.createVariable("nx", "f4", ("nx",))
xs.standard_name = "projection_x_coordinate"
xs.long_name = "X Georeferenced Coordinate for each pixel count"
xs.units = "m"

ys = rootgrp.createVariable("ny", "f4", ("ny",))
ys.standard_name = "projection_y_coordinate"
ys.long_name = "Y Georeferenced Coordinate for each pixel count"
ys.units = "m"

times = rootgrp.createVariable("time", "f8", ("time", ))
times.standard_name = "time"
times.units = "seconds since 1970-01-01 00:00:00 +0000"

mean_acq_times = rootgrp.createVariable("mean_scanline_acquisition_time", "f8", ("time", "ny"))
mean_acq_times.standard_name = "time"
mean_acq_times.units = "nanoseconds since 1970-01-01 00:00:00 +0000"

lons = rootgrp.createVariable("lon", "f4", ("ny", "nx"), fill_value=np.inf)
lons.valid_range = [-180., 180.]
lons.standard_name = "longitude"
lons.long_name = "Longitude at the centre of each pixel"
lons.units = "degrees_east"

lats = rootgrp.createVariable("lat", "f4", ("ny", "nx"), fill_value=np.inf)
lats.valid_range = [-90., 90.]
lats.standard_name = "latitude"
lats.long_name = "Latitude at the centre of each pixel"
lats.units = "degrees_north"

imgs = {}
for product in variables:
    imgs[product] = rootgrp.createVariable(product, "u1", ("time", "rgb", "ny", "nx", ), fill_value=0)
    imgs[product].grid_mapping = "projection_geostationary"
    imgs[product].units = "1"

projs = rootgrp.createVariable("projection_geostationary", "u1")
projs.grid_mapping_name = "geostationary"
projs.proj4 = local_scene[variables[0]].attrs['area'].proj_str
projs.longitude_of_projection_origin = local_scene[variables[0]].attrs['area'].proj_dict['lon_0']
projs.perspective_point_height = local_scene[variables[0]].attrs['area'].proj_dict['h']
projs.semi_major_axis = local_scene[variables[0]].attrs['area'].proj_dict['a']
projs.inverse_flattening = local_scene[variables[0]].attrs['area'].proj_dict['rf']
projs.sweep_angle_axis = 'y'

# Global attributes
# MMD temporal_extent,start_date, ACDD time_coverage_start
# rootgrp.time_coverage_start = '{:%Y-%m-%dT%H:%M:%S.%f}Z'.format(local_scene[variables[0]].attrs['start_time'])
# rootgrp.time_coverage_end = '{:%Y-%m-%dT%H:%M:%S.%f}Z'.format(local_scene[variables[0]].attrs['end_time'])
rootgrp.time_coverage_start = '{:%Y-%m-%dT%H:%M:%S.%f}Z'.format(datetime.utcfromtimestamp((acq_time[-1] - unix_epoch) / one_second))
rootgrp.time_coverage_end = '{:%Y-%m-%dT%H:%M:%S.%f}Z'.format(datetime.utcfromtimestamp((acq_time[1] - unix_epoch) / one_second))
rootgrp.slot_time = '{:%Y-%m-%dT%H:%M:%S.%f}Z'.format(local_scene.attrs['start_time'])

# MMD platform short_name, ACDD platform
rootgrp.platform = pl_sn[local_scene[variables[0]].attrs['platform_name']]
rootgrp.platform_long_name = pl_ln[local_scene[variables[0]].attrs['platform_name']]
rootgrp.platform_vocabulary = pr[local_scene[variables[0]].attrs['platform_name']]
rootgrp.platform_resource = pr[local_scene[variables[0]].attrs['platform_name']]

# MMD instrument short_name, ACDD instrument
rootgrp.instrument = instr_sn[local_scene[variables[0]].attrs['sensor']]
rootgrp.instrument_long_name = instr_ln[local_scene[variables[0]].attrs['sensor']]
rootgrp.instrument_vocabulary = ir[local_scene[variables[0]].attrs['sensor']]
rootgrp.instrument_resource = ir[local_scene[variables[0]].attrs['sensor']]
rootgrp.ancillary_timeliness = 'NRT'

rootgrp.date_created = '{:%Y-%m-%dT%H:%M:%S}Z'.format(datetime.now(tz=timezone.utc))

rootgrp.date_metadata_modified = '{:%Y-%m-%dT%H:%M:%S}Z'.format(datetime.now(tz=timezone.utc))
rootgrp.date_metadata_modified_type = 'Created'
rootgrp.date_metadata_modified_note = ''

# Find geospatial min and max longitude and latitude
sub_lat = latitude[start_y:end_y, start_x:end_x]
sub_lon = longitude[start_y:end_y, start_x:end_x]

rootgrp.geospatial_lat_min = sub_lat[np.isfinite(sub_lat)].min()
rootgrp.geospatial_lat_max = sub_lat[np.isfinite(sub_lat)].max()
rootgrp.geospatial_lon_min = sub_lon[np.isfinite(sub_lon)].min()
rootgrp.geospatial_lon_max = sub_lon[np.isfinite(sub_lon)].max()

# Assing data to each variable
rgbs[:] = [0, 1, 2]
xs[:] = pxc[start_x:end_x]
ys[:] = pyc[start_y:end_y]

lons[:] = longitude[start_y:end_y, start_x:end_x]
lats[:] = latitude[start_y:end_y, start_x:end_x]

times[:] = [(local_scene.attrs['start_time'] - datetime(1970, 1, 1)).total_seconds()]

# There is a bug in satpy not lfipping the coords correctly. Need to flip here.
# Fix underway in PR https://github.com/pytroll/satpy/pull/1600
mean_acq_times[0, :] = acq_time

for product in variables:
    print("Doing product:", product)
    imgs[product][0, 0, :] = data_enh[product].data[0, start_y:end_y, start_x:end_x].clip(0, 1) * 254. + 1
    # As the water vapur 62 is a single channel I copy this to all RGB to get the dimmension correct
    if product == 'water_vapour_62':
        imgs[product][0, 1, :] = data_enh[product].data[0, start_y:end_y, start_x:end_x].clip(0, 1) * 254. + 1
        imgs[product][0, 2, :] = data_enh[product].data[0, start_y:end_y, start_x:end_x].clip(0, 1) * 254. + 1
        continue
    imgs[product][0, 1, :] = data_enh[product].data[1, start_y:end_y, start_x:end_x].clip(0, 1) * 254. + 1
    imgs[product][0, 2, :] = data_enh[product].data[2, start_y:end_y, start_x:end_x].clip(0, 1) * 254. + 1

# Closing and saving!
rootgrp.close()
