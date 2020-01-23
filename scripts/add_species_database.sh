#!/bin/bash
# bash script to add Tag species to species_database_mod.F90

string1="CO2TAG"

echo "How many tag species do you want to add?"
read nm_tags

# loop over number of tags, add them to species_database_mod.F90, following the syntax
for (( counter=nm_tags; counter>0; counter-- ))
do
echo -n "                CASE( '$string1$counter' )\n                   FullName = 'Carbon dioxide from $string1$counter'"
# following line adds a line 'Species name ...' between 23rd and 24th line of input.geos
sed -i "4537i\                CASE( '$string1$counter' )\n                   FullName = 'Carbon dioxide from $string1$counter'" species_database_mod.F90
printf "\n"
string_paste+=", '$string1$counter' "
done

echo "Added lines above before 4537th line of species_database_mod.F90"

echo "Please manually add following lines to aroung line 4510, into CASE():"




