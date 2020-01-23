#!/bin/bash
# bash script to add Tag species to input.geos and HEMCO_Config.rc

# Some species we could add by default, to CO2 simulations
#sed -i "24i\Species name            : CO2ff" input.geos
#sed -i "24i\Species name            : CO2bf" input.geos
#sed -i "24i\Species name            : CO2oc" input.geos

# add tag species as well, first input number of tags you would like:
echo "How many tag species do you want to add?"
read nm_tags

echo "Would you like to modify input.geos: 1 for yes, 0 for no:"
read input_flag

# modify input.geos
if [ $input_flag -eq 1 ];
then

# loop over number of tags, add them to input.geos, following the syntax
for (( counter=nm_tags; counter>0; counter-- ))
do
echo -n "Species name            : $string1$counter"
# following line adds a line 'Species name ...' between 23rd and 24th line of input.geos
sed -i "28i\Species name            : $string1$counter" input.geos
printf "\n"
done

echo "Added lines above between 27th and 28th line of input.geos"

fi
# done modifying input.geos

echo "Would you like to modify HEMCO_Config? type 1 for yes, 0 for no:"
read hemco_flag

# modify hemco_config file
if [ $hemco_flag -eq 1 ];
then

if [ $nm_tags -gt 9 ];
then
echo "tag number is greater than 9"

# for numbers smaller than 10

for (( counter=$nm_tags; counter>9; counter-- ))
do
sed -i "145 i\0 $string1$counter  /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2019-12/EDGAR_v5_CO2_1970-2018.0.1x0.1.nc  emi_co2 1970-2018/1/1/0 C xy kg/m2/s $string1$counter 20$counter 40/41/42/80 1 2" HEMCO_Config.rc

((line_nm2 = 825 + $nm_tags - counter + 1))
echo $line_nm2
sed -i "$line_nm2 i\20$counter MASK$counter     /geos/u73/msadiq/GEOS-Chem/rundirs/geosfp_2x25_CO2/MASKS/MASK${counter}_1x1.nc   MASK 2000/1/1/0 C xy 1 1 -180/-90/180/90 " HEMCO_Config.rc
done

for (( counter=9; counter>0; counter-- ))
do
sed -i "145 i\0 $string1$counter  /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2019-12/EDGAR_v5_CO2_1970-2018.0.1x0.1.nc  emi_co2 1970-2018/1/1/0 C xy kg/m2/s $string1$counter 200$counter 40/41/42/80 1 2" HEMCO_Config.rc

((line_nm2 = 825 + 9 - counter + $nm_tags - 9 + 1))
echo $line_nm2
sed -i "$line_nm2 i\200$counter MASK$counter     /geos/u73/msadiq/GEOS-Chem/rundirs/geosfp_2x25_CO2/MASKS/MASK${counter}_1x1.nc   MASK 2000/1/1/0 C xy 1 1 -180/-90/180/90 " HEMCO_Config.rc
done

else
for (( counter=1; counter<$nm_tags; counter++ ))
do
sed -i "145 i\0 CO2Tag$counter  /geos/u73/msadiq/GEOS-Chem/HEMCO/CO2/v2019-12/EDGAR_v5_CO2_1970-2018.0.1x0.1.nc  emi_co2 1970-2018/1/1/0 C xy kg/m2/s $string1$counter 200$counter 40/41/42/80 1 2" HEMCO_Config.rc

((line_nm2 = 825 + counter))
echo $line_nm2
sed -i "$line_nm2 i\200$counter MASK$counter     /geos/u73/msadiq/GEOS-Chem/rundirs/geosfp_2x25_CO2/MASKS/MASK${counter}_1x1.nc   MASK 2000/1/1/0 C xy 1 1 -180/-90/180/90 " HEMCO_Config.rc
done

fi
fi


