#!/bin/bash

MISSION_DIRECTORY="../moos-ivp-younge/missions/ufld_saxis"
SHORE_DATA_DIRECTORY="./RAW_SHORE_DATA"
VEHICLE_DATA_DIRECTORY="./RAW_VEHICLE_DATA"
DEPOT_DATA_DIRECTORY="./RAW_DEPOT_DATA"
rm "$SHORE_DATA_DIRECTORY"/*
rm "$VEHICLE_DATA_DIRECTORY"/*
rm "$DEPOT_DATA_DIRECTORY"/*
for entry in "$MISSION_DIRECTORY"/XLOG*/*.alog
do
  #echo $entry
  cp $entry "$SHORE_DATA_DIRECTORY"
done

for entry in "$MISSION_DIRECTORY"/LOG*/*.alog
do
  if ! [[ "$entry"  =~ .*"DEPOT".* ]]; then
    #echo $entry
    cp $entry "$VEHICLE_DATA_DIRECTORY"
  fi
done

for entry in "$MISSION_DIRECTORY"/LOG*/*.alog
do
  if [[ "$entry"  =~ .*"DEPOT".* ]]; then
    echo $entry
    cp $entry "$DEPOT_DATA_DIRECTORY"
  fi
done