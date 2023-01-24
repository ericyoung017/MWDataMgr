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
NUM_DEPOTS=1
TANK_SIZE=50
#iterate through the arguments, if you see a -d, then the next argument is the number of depots. if you see a -t, then the next argument is the tank size
while getopts ":d:t:" opt; do
  case $opt in
    d) NUM_DEPOTS="$OPTARG"
    ;;
    t) TANK_SIZE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done
echo "Number of depots: $NUM_DEPOTS"
echo "Tank size: $TANK_SIZE"

#python3 MWDataMgr.py -s $entry -o output_files -i $VEHICLE_TOPICS --vehicle --moos --topic_mapping
rm "$SHORE_DATA_DIRECTORY"/*
rm "$VEHICLE_DATA_DIRECTORY"/*
rm "$DEPOT_DATA_DIRECTORY"/*
#rm -r "$OUTPUT_DIRECTORY"/*
#create an output directory string with the number of depots and tank size in it
OUTPUT_DIRECTORY="$OUTPUT_DIRECTORY/depots_$NUM_DEPOTS-tank_$TANK_SIZE"
echo "Output directory: $OUTPUT_DIRECTORY"
for entry in "$MISSION_DIRECTORY"/XLOG*/*.alog
do
  #echo $entry
  cp $entry "$SHORE_DATA_DIRECTORY"
  python3 ./MWDataMgr.py -s $entry -o $OUTPUT_DIRECTORY -i $SHORE_TOPICS --shore --moos --script --topic_mapping $SHORE_TOPICS_MAPPING &
done

for entry in "$MISSION_DIRECTORY"/LOG*/*.alog
do
  if ! [[ "$entry"  =~ .*"DEPOT".* ]]; then
    #echo $entry
    cp $entry "$VEHICLE_DATA_DIRECTORY"
    python3 ./MWDataMgr.py -s $entry -o $OUTPUT_DIRECTORY -i $VEHICLE_TOPICS --vehicle --moos --script --topic_mapping $VEHICLE_TOPICS_MAPPING &
  fi
done

for entry in "$MISSION_DIRECTORY"/LOG*/*.alog
do
  if [[ "$entry"  =~ .*"DEPOT".* ]]; then
    #echo $entry
    cp $entry "$DEPOT_DATA_DIRECTORY"
    python3 ./MWDataMgr.py -s $entry -o $OUTPUT_DIRECTORY -i $DEPOT_TOPICS --depot --moos --script --topic_mapping $DEPOT_TOPICS_MAPPING &
  fi
done
wait
echo "Done"