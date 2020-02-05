#!/bin/bash

# shell script to automatically resubmit geos-chem simulation after every month of simulation
# settings
first_month=1
last_month=3

for (( imonth=$first_month; imonth<=$last_month; imonth++ ))
do
echo "Simulation begins for month: $imonth"
# replace start and end dates in input.geos
sed -i "3c\Start YYYYMMDD, hhmmss  : 20180${imonth}01 000000" input.geos
sed -i "4c\End   YYYYMMDD, hhmmss  : 20180$((imonth+1))01 000000" input.geos

### start geos-chem simulation
./geos > log.run.month$imonth &

valid=true
while [ $valid ]
do
# count numbers of following string, if it is 1, simulate the following month, else, wait for another 100 seconds
var=$(grep -c "E N D   O F   G E O S -- C H E M" log.run.month$imonth)

if [ $var -eq 1 ];
then
echo "Simulation ended"
echo $(date)
echo "continue to run another month"
break # break while, start next simulation

else
echo "Simulation is going on..."
sleep 100
fi

done # while

done # for


