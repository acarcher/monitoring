#!/usr/bin/python3

import numpy as np
import glob


class nputil():

    def npmerge(directory, dimensions, sampletype, variety=False):
        # positive = list of files with positive samples in them
        # negative = ""

        print("Merging together features...")
        paths = glob.glob(directory + "/{}*.npy".format(sampletype))

        merged = np.empty(dimensions)
        max_rows = merged.shape[0]
        index = 0

        if variety:
            variety_rows, extra_rows = divmod(merged.shape[0], len(paths))

        for path in paths: # Need to handle the case where the feature vector is smaller than the specified number of samples

            partial_sample = np.load(path)

            if variety and variety_rows <= partial_sample.shape[0]:
                num_rows = variety_rows
            else:
                num_rows = partial_sample.shape[0]

            if index + num_rows > max_rows:
                num_rows = max_rows - index - 1

            merged[list(range(index, index + num_rows)), :] = partial_sample[list(range(0, num_rows)), :]
            index = index + num_rows if index != 0 else index + num_rows - 1

            #print(index)
            #print(dimensions[0])

            if variety and paths.index(path) + 1 == len(paths) and \
                num_rows + extra_rows < partial_sample.shape[0]: # last item, time to correct juicy rounding errors

                merged[list(range(index, index + extra_rows)), :] = \
                    partial_sample[list(range(num_rows, num_rows + extra_rows)), :]

            if (index + 1) % dimensions[0] == 0:
                print("Done merging features")
                return merged

        return merged
