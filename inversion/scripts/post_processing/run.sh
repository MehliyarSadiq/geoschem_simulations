#!/bin/bash

year=2016

for month in {3..12}
do
    # 1, run the model
    # 2 hours
    #cd /geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs/$year/CO2-$year
    #./geos > log.run

    # 1.1, copy .dat files into the directory

    # 2, post-processing
    # python script
    # python post_process.py $month
    # < 0.5 hour
    
    # 2.1, inverse calculation
    # python inversion.py $month

    # 3, update time period, update output directory
    


    # add 0 before the month string if it is smaller than 10
    if [ $month -gt 9 ];
    then
    month_string="$month"
    else
    month_string="0$month"
    fi
    
    # case directory
    #cd /geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs
    #cd CO2-TC67-$year-$month_string
    #pwd

    ## run the simulation
    #./geos > log.run 

done

