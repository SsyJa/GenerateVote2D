import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# generate centroids for each masks in seg_map(SAM mask_l)

seg_folder = ""
save_folder = ""

sam_index = 3

def count_npy_files(path):
    floder = Path(path)
    npy_files = [f.name for f in floder.glob('*_s.npy')]
    npy_files = sorted(npy_files)
    return sum(1 for item in floder.glob('*_s.npy') if item.is_file()), npy_files

def calculate_centroids_by_label(npy_count, npy_list, save_centroids_floder):

    save_dic_folder = os.path.join(save_centroids_floder, "dics")
    os.makedirs(save_dic_folder, exist_ok= True)
    save_cen_folder = os.path.join(save_centroids_floder, "Cen")
    os.makedirs(save_cen_folder, exist_ok= True)

    for i in tqdm(range(npy_count), desc='Processing'):
        seg_path = seg_folder + '/' + npy_list[i]
        seg_map = np.load(seg_path) # 4, 1080, 1440
        seg_map = seg_map[sam_index] # we choose mask_l from SAM

        unique_labels = np.unique(seg_map)
        unique_labels = unique_labels[unique_labels != -1]

        centroids_dic = {}
        d_map = np.full((2, seg_map.shape[0], seg_map.shape[1]), -1, dtype=np.float32)

        rows, cols = seg_map.shape # 1080, 1440
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))

        for label in unique_labels:
            mask = (seg_map == label) # bool matrix
            intensity = mask.astype(np.int32)
            total_intensity = np.sum(intensity)
            if total_intensity == 0:
                centroids_dic[label] = None
                continue
    # centroid
            x_centroid = np.sum(x_coords * intensity) / total_intensity
            y_centroid = np.sum(y_coords * intensity) / total_intensity
            x_centroid = int(round(x_centroid))
            y_centroid = int(round(y_centroid))

            centroids_dic[label] = (x_centroid, y_centroid)  # float

            d_map[0] = np.where(mask, x_centroid, d_map[0])
            d_map[1] = np.where(mask, y_centroid, d_map[1])

    # save        
        seg_path_name = Path(seg_path)
        save_dic_name = seg_path_name.stem + '_cen_dic' + seg_path_name.suffix
        save_dic_path = save_dic_folder + '/' + save_dic_name
        np.save(save_dic_path, centroids_dic)

        save_cen_name = seg_path_name.stem + '_cen' + seg_path_name.suffix
        save_cen_path = save_cen_folder + '/' + save_cen_name
        np.save(save_cen_path, d_map)
    


if __name__ == '__main__':
    npy_count, npy_list = count_npy_files(seg_folder)
    npy_count = int(npy_count)

    save_centroids_floder = os.path.join(save_folder, "centroids")
    os.makedirs(save_centroids_floder, exist_ok = True)

    calculate_centroids_by_label(npy_count, npy_list, save_centroids_floder)
