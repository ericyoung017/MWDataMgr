#!/bin/bash

target_dir=$1

#iterate through every folder in the post process directory, for each folder, count the number of files whose name contains the string "XLOG" and output this number and the folder name to ther terminal
for folder in "$target_dir"/*
do
  echo $folder
  #count the number of files whose name contains the string "XLOG"
  num_xlogs=$(find $folder -name "*XLOG*" | wc -l)
  #echo the num_xlog to a new line
  echo -e "$num_xlogs"
done
