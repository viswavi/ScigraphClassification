# run in python2.7
'''
python2.7 convert_planetoid_data_to_python3.py \
        --planetoid-directory /projects/ogma1/vijayv/planetoid/data \
        --new-planetoid-directory /projects/ogma1/vijayv/planetoid/data3

'''
import argparse
import cPickle
import json
import scipy.sparse
import shutil
import numpy as np
import os

def convert_numpy(array, outfile):
    np.savetxt(outfile + ".csv", array)

def convert_scipy(matrix, outfile):
    scipy.sparse.save_npz(outfile + ".npz", matrix)

def convert_graph(graph, outfile):
    json.dump(graph, open(outfile + ".json", 'w'))

def convert_all_files(old_dir, new_dir):
    files = os.listdir(old_dir)
    for f in files:
        if f.endswith("index"):
            shutil.copyfile(os.path.join(old_dir, f), os.path.join(new_dir, f) + ".csv")
        else:
            try:
                data = cPickle.load(open(os.path.join(old_dir, f)))
            except:
                import pdb; pdb.set_trace()
            if f.endswith(".x") or f.endswith(".tx") or f.endswith(".allx"):
                convert_scipy(data, os.path.join(new_dir, f))
            elif f.endswith(".y") or f.endswith(".ty") or f.endswith(".ally"):
                convert_numpy(data, os.path.join(new_dir, f))
            elif f.endswith(".graph"):
                convert_graph(data, os.path.join(new_dir, f))
            else:
                raise ValueError("Unsupported file type {}".format(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planetoid-directory', type=str, required=False, default="/projects/ogma1/vijayv/planetoid/data")
    parser.add_argument('--new-planetoid-directory', type=str, required=False, default="/projects/ogma1/vijayv/planetoid/data3")
    args = parser.parse_args()
    if not os.path.exists(args.new_planetoid_directory):
        os.makedirs(args.new_planetoid_directory)
    planetoid_data = convert_all_files(args.planetoid_directory, args.new_planetoid_directory)

if __name__ == "__main__":
    main()
