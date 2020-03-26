#!/bin/bash

# script to create simulation, compile and run for a month
# settings
month=5
# remove existing directory, will be created in this script
rm -rf geosfp_2x25_CO2

echo "Simulation begins for month: $month"

# add 0 before the month string if it is smaller than 10
if [ $month -gt 9 ];
then
month_string="$month"
month_string_p1="$month+1"

else
month_string="0$month"
month_string_p1="0$((month+1))"
fi

# change settings in Unit Testers to create a brand new simulation
cd /geos/u73/msadiq/GEOS-Chem/UT/perl
sed -i "111c\geosfp   2x25        -      CO2              2018${month_string}0100   2018${month_string_p1}0100     -" CopyRunDirs.input
# create a brand new case
./gcCopyRunDirs

# case directory
cd /geos/u73/msadiq/GEOS-Chem/rundirs/ensemble_runs
mv geosfp_2x25_CO2 CO2-2018-$month_string
cd CO2-2018-$month_string
pwd

# configuration settings
# 1, input.geos
cp ../input.geos ./
# replace start and end dates
sed -i "3c\Start YYYYMMDD, hhmmss  : 2018${month_string}01 000000" input.geos
sed -i "4c\End   YYYYMMDD, hhmmss  : 2018${month_string_p1}01 000000" input.geos

# 2, HEMCO_Config
cp ../HEMCO_Config.rc ./
# replace mask settings
sed -i "s/2018-01/2018-$month_string/g" HEMCO_Config.rc

# 3, HISTORY.rc
cp ../HISTORY.rc ./


## compile the model
make realclean; make -j4 build

### start geos-chem simulation
./geos > log.run.month$month_string &

valid=true
while [ $valid ]
do
# count numbers of following string, if it is 1, simulation ends, else, wait for another 100 seconds
var=$(grep -c "E N D   O F   G E O S -- C H E M" log.run.month$month_string)

if [ $var -eq 1 ];
then
echo "Simulation ended"
echo $(date)
break # break while, end

else
echo "Simulation is going on..."
sleep 300
fi

done # while

