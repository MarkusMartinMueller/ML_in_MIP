import argparse
from pathlib import Path
import numpy as np
import json
import nibabel as nib
from tqdm import tqdm
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description='Converting masks to non axis-aligned bounding boxes.')
    parser.add_argument('-path', metavar='-p', type=str)

    return parser.parse_args()


def create_bounding_box(image, label, affine):

    # PCA
    voxel_coordinates = np.array(np.where(image == label)).T
    voxel_coordinates_ext = np.concatenate((voxel_coordinates, np.ones((voxel_coordinates.shape[0], 1))), axis=1)
    world_coordinates = (affine @ voxel_coordinates_ext.T).T[:, :3]

    # mask_ixs = np.array(np.where(image == label)).T
    pca = PCA(n_components=3)
    mask_ixs_trans = pca.fit_transform(world_coordinates)

    # bounding box
    components = pca.components_.T
    bb_extent = np.max(mask_ixs_trans, axis=0) - np.min(mask_ixs_trans, axis=0)
    bb_center = pca.inverse_transform(np.min(mask_ixs_trans, axis=0) + bb_extent / 2)

    return {
        'position': bb_center.tolist(),
        'object_oriented_bounding_box': {
            'extent': bb_extent.tolist(),
            'orthogonal_offset_vectors': components.tolist()
        }
    }


def create_bounding_boxes(image, affine):

    #
    bboxes = []
    for label in np.unique(image)[1:]:
        bboxes.append(create_bounding_box(image, label, affine))

    return bboxes


def main():

    # get arguments
    args = parse_args()
    files = Path(args.path).glob('*labeledMasks.nii.gz')

    # load masks
    task_1_results = []
    for file in tqdm(list(files)):

        #
        dataset_id = file.stem.replace('_labeledMasks.nii', '')
        image = nib.load(file)
        image_np = image.get_fdata()

        #
        candidates = create_bounding_boxes(image_np, image.affine)

        #
        task_1_results.append({
            'dataset_id': dataset_id,
            'processing_time_in_seconds': 22,
            'candidates': candidates
        })

    # write to file
    result = {
        'grand_challenge_username': 'ICM',
        "used_hardware_specification": {
            "CPU": "Intel Core i9 9900K 8x 3.60GHz",
            "GPU": "NVIDIA RTX 2080 Ti",
            "#GPUs": 1,
            "RAM_in_GB": 4,
            "additional_remarks": "special hardware requirements, other comments"
        },
        'task_1_results': task_1_results
    }
    with open(str(Path(args.path) / 'reference.json'), 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
