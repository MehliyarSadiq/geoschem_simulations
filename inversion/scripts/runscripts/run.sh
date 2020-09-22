#!/bin/bash
# script to run GEOS-Chem full CO2 simulation continuously
# requires tag CO2 runs to finish

year=2016

for month in {11..12}
do
    cd /scratch/local/msadiq/rundir/2016/runscripts/
    
    # 1, process previous runs
    # python script
    python inversion_ultimate_script.py $month

    # 2, update fluxes, update time period, update output directory
    # add 0 before the month string if it is smaller than 10
    month_begin=$((month-2))
    if [ $month_begin -lt 10 ];
    then
    month_string="0$month_begin"
    else
    month_string="$month_begin"
    fi

    month_last=$((month+2))    
    if [ $month_last -lt 10 ];
    then
    month_string_last="0$month_last"
    elif [ $month_last -gt 12 ];
    then
    month_string_last="0$((month_last-12))"
    else
    month_string_last="$month_last"
    fi
 
    #echo ${year}${month_string}
    #echo ${year}${month_string_last}
    
    cd /scratch/local/msadiq/rundir/$year/CO2-$year
    
    # configuration settings
    # 1, input.geos
    # replace start and end dates
    sed -i "3c\Start YYYYMMDD, hhmmss  : ${year}${month_string}01 000000" input.geos
    sed -i "4c\End   YYYYMMDD, hhmmss  : ${year}${month_string_last}01 000000" input.geos

    # change hemco config file
    month_names=("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    new_month=${month_names[$((month-1))]%,}
    new_month_p1=${month_names[$month]%,}
    #echo "173c\0 OCEANCO2_TAKA_MONTHLY  /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/Taka2009_CO2_Monthly.updated.2016.${new_month}.nc     CO2 2000/1-12/1/0      C xy kg/m2/s CO2   - 2 2"
    #echo "200c\0 SIB_BBIO_CO2    /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/NEE.Liang.2016.updated.2016.${new_month}.nc CO2 2016/1-12/1-31/0-23 C xy kg/m2/s CO2    - 3 1"
    
    sed -i "173c\0 OCEANCO2_TAKA_MONTHLY  /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/Taka2009_CO2_Monthly.updated.2016.${new_month}.nc     CO2 2000/1-12/1/0      C xy kg/m2/s CO2   - 2 2" HEMCO_Config.rc
    sed -i "200c\0 SIB_BBIO_CO2    /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2020-04/NEE.Liang.2016.updated.2016.${new_month}.nc CO2 2016/1-12/1-31/0-23 C xy kg/m2/s CO2    - 3 1" HEMCO_Config.rc
    sed -i "258c\LT avg timeseries file  : ./nd51/${new_month_p1}/ts_satellite.YYYYMMDD.bpch" input.geos 
    # run the model
    ./geos > log.run

done

