import pandas as pd
import pyarrow
#

import os
#declare a main function
if __name__ == '__main__':
    input_dir = './output_files'
    output_dir = './FeatherFiles'
    #get the total amount of files in the input directory recursively that are not .DS_Store
    total_files = sum([len(files) for r, d, files in os.walk(input_dir) if not files[0] == '.DS_Store'])

    #iterate recursively through the input directory and read all the pickle files containing the "_" character to a pandas dataframe
    #for each pickle file, write the dataframe to a feather file in the same folder structure as the input directory
    counter= 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if "DS_Store" not in file:
                input_file = os.path.join(root, file)
                print("reading file"+input_file)
                #print("output_dir: " + output_dir)
                #print("replaced: " + root.replace(input_dir, ''))


                output_file = os.path.join(root.replace(input_dir, output_dir),file)
                #print("Reading pickle file: " + input_file)
                df = pd.read_pickle(input_file)
                #print("Writing feather file: " + output_file)
                df.to_feather(input_file)
                print("wrote file"+ str(counter) + " of " + str(total_files))
                counter += 1
                #feather.write_dataframe(df, output_file)