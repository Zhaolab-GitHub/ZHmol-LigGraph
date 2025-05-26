import numpy as np
import os
import sys
import multiprocessing as mp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cv", help="cross-validation fold", type=int, default = 0)
parser.add_argument("--resolution", help="resolution of the pose", type=int, default = 20)
parser.add_argument("--bond_th", help="create a bond for the pair of atoms which distance less than bond_th", type=int, default = 6)
parser.add_argument("--pocket_th", help="the threshold of distance to the centroid of ligand to be considered as pocket", type=float, default = 12)
parser.add_argument("--output_file", help="output file name of this train/test data", type=str, default = None)
parser.add_argument("--start_iter", help="create training data from which random seed", type=int, default = 3)
parser.add_argument("--end_iter", help="create training data till which random seed", type=int, default = 14)
parser.add_argument("--thread_num", help="num of threads to creating dataset", type=int, default = 4)
parser.add_argument("--use_new_data", help="create data for predicting 3D coordinate", default=False, action='store_true')
parser.add_argument("--screen_data", help="If we generate data for screen", default=False, action='store_true')
parser.add_argument("--pdbbind_dir", help="dir to the pdbbind dataset output", type=str, default = None)
parser.add_argument("--label_list_file", help="the path to label files", type=str, default = None)
parser.add_argument("--input_list", help="list of train/test pdbs", type=str, default = None)
parser.add_argument("--groundtruth_dir", help="the path to the ground truth pose pdbbind files", type=str)
parser.add_argument("--dataset", help="type of dataset", type=str, default='coor2')
parser.add_argument("--muv_dir", help="path to muv dataset", type=str, default = None)
parser.add_argument("--muv_target", help="target id of muv data", type=str, default='466')
parser.add_argument("--muv_label", help="which label of the muv data", type=str, default='decoy')
parser.add_argument("--casf_groundtruth_dir", help="path to casf dataset GT", type=str, default = None)
parser.add_argument("--casf_dir", help="path to casf dataset", type=str, default = None)
args = parser.parse_args()


def srand_data_load_save_coord2_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = []
    groundtruth_suffix = ['.rec.pdb', '.lig.mol2']

    if args.dataset == 'coor2':
        import coordinate_data2 as cd2
        splits = ['train', 'test']
        print("import coordinate_data2")

    for split in splits:
        input_list_filename = input_list + split
        with open(input_list_filename, 'r') as gf:
            inputs = gf.readlines()
            start = (thread_id * len(inputs)) // thread_num
            end = ((thread_id + 1) * len(inputs)) // thread_num
        cd2.read_pdbbind_to_disk_rmsd_energy_split(input_list_filename,
                                                   groundtruth_dir,
                                                   groundtruth_suffix,
                                                   pdbbind_dir,
                                                   output_dir_tmp+'/' + split,
                                                   resolution,
                                                   tile_size,
                                                   bond_th,
                                                   pocket_th,
                                                   start, end)
    
def srand_data_load_save_coord2(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_coord2")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    if thread_num == 1:
        srand_data_load_save_coord2_thread(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, 1, 0)
    else:
        p_list = []
        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_coord2_thread,
                           args=(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num, thread_id))
            p.start()
            p_list.append(p)
    
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        for split in ['train', 'test']:
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            n = len(dataset_file_list) // 3
            file_num = len(os.listdir(output_dir+'/'+split)) // 3
            for j in range(file_num, file_num + n):
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-feats'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-feats')
                dataset = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_data-G.json'
                os.rename(dataset, output_dir+'/'+split+'/'+str(j)+'_data-G.json')
                label = output_dir_tmp+'/'+split+'/'+str(j-file_num)+'_label'
                os.rename(label, output_dir+'/'+split+'/'+str(j)+'_label')
                
if __name__ == "__main__":
    cv = args.cv
    label_list_file = args.label_list_file
    pdbbind_dir = args.pdbbind_dir
    
    groundtruth_dir = args.groundtruth_dir
    iteration = args.resolution
    resolution = iteration
    bond_th = args.bond_th
    pocket_th = args.pocket_th
    output_file = args.output_file
    output_dir = label_list_file + '/' + output_file
    input_list = label_list_file + '/pdb_list_'
    input_list = args.input_list
    casf_groundtruth_dir = args.casf_groundtruth_dir
    casf_dir = args.casf_dir
    start = args.start_iter
    end = args.end_iter
    thread_num = args.thread_num
    use_new = args.use_new_data
    screen_data = args.screen_data
    print(cv, iteration, output_file, start, end, thread_num, use_new)
    srand_data_load_save_coord2(input_list, groundtruth_dir, pdbbind_dir, output_dir, resolution, bond_th, pocket_th, iteration, thread_num = thread_num)

