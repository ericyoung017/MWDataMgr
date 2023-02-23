#!/bin/bash

MISSION_DIRECTORY="../moos-ivp-younge/missions/ufld_saxis"
SHORE_DATA_DIRECTORY="./RAW_SHORE_DATA"
VEHICLE_DATA_DIRECTORY="./RAW_VEHICLE_DATA"
DEPOT_DATA_DIRECTORY="./RAW_DEPOT_DATA"
VEHICLE_TOPICS="./testing_space/moos_topic_ix_vehicle.cfg"
VEHICLE_TOPICS_MAPPING="./testing_space/moos_topic_mapping_vehicle.cfg"
DEPOT_TOPICS="./testing_space/moos_topic_ix_depot.cfg"
DEPOT_TOPICS_MAPPING="./testing_space/moos_topic_mapping_depot.cfg"
SHORE_TOPICS="./testing_space/moos_topic_ix_shore.cfg"
SHORE_TOPICS_MAPPING="./testing_space/moos_topic_mapping_shore.cfg"
OUTPUT_DIRECTORY="./output_files"
POST_PROCESS_DIRECTORY="../postprocess"
NUM_DEPOTS=1
TANK_SIZE=50
rm "$SHORE_DATA_DIRECTORY"/*
rm "$VEHICLE_DATA_DIRECTORY"/*
rm "$DEPOT_DATA_DIRECTORY"/*
#iterate through every folder in the post process directory. split each folder name by the underscore, and then get the number of depots, number of ships, and tank size from the folder name
for folder in "$POST_PROCESS_DIRECTORY"/*
do
  #echo $entry
  #split the folder name by the underscore
  IFS='_' read -ra ADDR <<< "$folder"
  #get the number of depots from the folder name
  NUM_DEPOTS="${ADDR[1]}"
  #get the tank size from the folder name
  TANK_SIZE="${ADDR[5]}"
  #get the number of ships from the folder name
  NUM_SHIPS="${ADDR[3]}"
#XLOGS are way slower than the LOGS, so we will process them first in parallel to maximize the speed of the script
  UNIQUE_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/depots_$NUM_DEPOTS-tank_$TANK_SIZE-ships_$NUM_SHIPS"
  echo "Output directory: $UNIQUE_OUTPUT_DIRECTORY"
  for entry in "$folder"/XLOG*/*.alog
  do
    #echo $entry
    cp $entry "$SHORE_DATA_DIRECTORY"
    python3 ./MWDataMgr.py -s $entry -o $UNIQUE_OUTPUT_DIRECTORY -i $SHORE_TOPICS --shore --moos --script --topic_mapping $SHORE_TOPICS_MAPPING &
  done

done
wait
#Now that the XLOGS are processed, we can process the LOGS in parallel
for folder in "$POST_PROCESS_DIRECTORY"/*
do
  #echo $entry
  #split the folder name by the underscore
  IFS='_' read -ra ADDR <<< "$folder"
  #get the number of depots from the folder name
  NUM_DEPOTS="${ADDR[1]}"
  #get the tank size from the folder name
  TANK_SIZE="${ADDR[5]}"
  #get the number of ships from the folder name
  NUM_SHIPS="${ADDR[3]}"
  #create an output directory string with the number of depots and tank size in it
  UNIQUE_OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/depots_$NUM_DEPOTS-tank_$TANK_SIZE-ships_$NUM_SHIPS"
  echo "Output directory: $UNIQUE_OUTPUT_DIRECTORY"

  for entry in "$folder"/LOG*/*.alog
  do
    if ! [[ "$entry"  =~ .*"DEPOT_".* ]]; then
      #echo $entry
      cp $entry "$VEHICLE_DATA_DIRECTORY"
      echo -e $entry
      python3 ./MWDataMgr.py -s $entry -o $UNIQUE_OUTPUT_DIRECTORY -i $VEHICLE_TOPICS --vehicle --moos --script --topic_mapping $VEHICLE_TOPICS_MAPPING &
    fi
  done

  for entry in "$folder"/LOG*/*.alog
  do
    if [[ "$entry"  =~ .*"DEPOT_".* ]]; then
      #echo $entry
      cp $entry "$DEPOT_DATA_DIRECTORY"
      python3 ./MWDataMgr.py -s $entry -o $UNIQUE_OUTPUT_DIRECTORY -i $DEPOT_TOPICS --depot --moos --script --topic_mapping $DEPOT_TOPICS_MAPPING &
    fi
  done
  wait
done