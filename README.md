# GEOS-Chem simulations

# Basics:

1. Get access to group machines
- machines: bosch, poirot, renjie or rebus. login to my Linux machine first, in the terminal, then type ' ssh machine_name'
- how to simplify login process and set up the environment? Yao Fei has a detailed page on this: https://github.com/FeiYao-Edinburgh/PhD-Edinburgh/blob/master/Documentation/setupComEnv.md 
- met data for GC is stored in: **/geos/d21/GC_DATA/ExtData (download if you need anything else)**
- HEMCO data is stored in: **/geos/d21/GC_DATA/ExtData/HEMCO (download if missing)**
- working directory: **/geos/u73/msadiq**

2.  Getting started with GEOS-Chem

- youtube video: https://www.youtube.com/watch?v=LOobuuhNdAg&t=1558s
- setup linux environment (e.g. compilers, NetCDF libs, etc.), M sent me his settings in .bashrc script. Fei also has .bashrc settings on his Github. Both are similar.
- Youtube video's building part is outdated, check GC wiki for updated one: make -j4 build, instead of make -j4 mpbuild
- and then run the model.

3. Simplified steps to run a simulation, after downloading source code and UT:

1. go to UT directory, modify CopyRunDirs.input (code version, code location, data location, simulation type, length)
2. execute gcCopyRunDirs
3. go to case directory, modify input.geos, HEMCO_Config.rc, HEMICO_Diag.rc, HISTORY.rc (freq, dur)
4. compile: make realclean; make -j4 build
5. run by: ./geos > log.xxxx
- if missing emission data, which I have, go to hemco_data_download, modify Sadiq_download.rc (copied from forTesting.rc), add emission you need following the format, and ./hemcoDataDownload.pl Sadiq_download.rc

## How-tos:

 - Log into renjie or other machines outside of campus:

ssh msadiq@xxxxxxxxx.ac.uk

ssh renjie

 - synchronising files remotely to your local machine:

rsync -avp /geos/u73/msadiq/analysis/new/plots  /home/msadiq/scratch

wget -r -np -nH -R "*.html" "http://geoschemdata.computecanada.ca/DIRECTORY_NAME"

 - quick-look at NetCDF files on Linux machine

ncview xx.nc

 - merge multiple NetCDF files: ncecat *nc -O [merge.nc](http://merge.nc)

[http://dvalts.io/data/modelling/2018/01/16/NetCDF-merging.html](http://dvalts.io/data/modelling/2018/01/16/NetCDF-merging.html)

 - scale a variable in NetCDF file: ncflint --fix_rec_crd -w 1000000,0 [GEOSChem.SpeciesConc.201607.nc](http://geoschem.speciesconc.201607.nc/) [GEOSChem.SpeciesConc.201607.nc](http://geoschem.speciesconc.201607.nc/) GEOSChem.SpeciesConc.201607_scaled.nc

 - check out NCO, really convenient tools there, to manipulate NetCDF files

## Workflow (machine: renjie/rebus)

- run global simulation (2x25 CO2), save boundary conditions (default: 3 hourly)
- set up nested-grid simulations
    - domain: FlexGrid, CHE definition, **11W-36E longitude and 36N-64N latitude**
    - Tran/conv timestep **600 → 300 (due to error message, seconded by M)**
    - met data: /geos/d21/GC_DATA/
    - HEMCO data: /geos/d21/HEMCO/

## Test run details:

- Test run: global methane 4x5 sim for 1 week
- Test run: global CO2 2x25 for 1 month → boundary conditions → preparing for nested grid simulation for Europe
- Nested-grid sim:
    - Domain: follow CHE, 11W-36E, 36N-64N (GC default: 15W-40E, 32.75N-61.25N)
    - Emissions: default
- Error messages:
    - ERROR: could not find restart file when copying case directory using UT. Mark helped solve the problem by creating a symbolic link in /geos/d21/GC_DATA
    - ERROR: cannot get field GFED_TEMP from /geos/d21/GC_DATA//HEMCO/GFED4/v2015-10/2017/GFED4_gen.025x025.201707.nc → change HEMCO_Config.rc: 111 GFED_TEMF       $ROOT/GFED4/v2015-10/$YYYY/GFED4_gen.025x025.$YYYY$MM.nc  DM_TEMF       1997-**2016**/1-12/01/0    RF xy kgDM/m2/s * - 1 1 → 111 GFED_TEMF $ROOT/GFED4/v2015-10/$YYYY/GFED4_gen.025x025.$[YYYY$MM.nc](http://yyyy%24mm.nc/) **DM_TEMF** 1997-**2018**/1-12/01/0 RF xy kgDM/m2/s * - 1 1, and all the following similar lines
    - ERROR: cannot find file: CO2_prod_rates.GEOS5.2x25.47L.nc, point to /geos/u73/GC_DATA/HEMCO/CO2/v2019-02/CO2_prod_rates.GEOS5.2x25.47L.nc
    - ERROR: Reading /geos/d21/GC_DATA/CHEM_INPUTS/FAST_JX/**v2019-06**/FJX_spec.dat and model stops → v2019-06 is missing, downloaded this to my own directory and pointed model to it.
