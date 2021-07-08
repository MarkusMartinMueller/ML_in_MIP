import json
from pathlib import Path

#
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_img
import pyvista as pv

#
from evalutils import DetectionEvaluation, ClassificationEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator
from evalutils.io import FileLoader
from typing import Dict


class JSONLoader(FileLoader):
    def __init__(self):
        super().__init__()

    def load(self, fname):

        if fname.suffix != '.json' or fname.stem == 'schema':
            return None

        with open(fname) as json_file:
            data = json.load(json_file)
            # TODO: validate json
            return pd.DataFrame(data['task_1_results'])


def create_bounding_box_pv(candidate):

    # create ground-truth bounding box
    bb_center = candidate['position']
    bb_extent = candidate['object_oriented_bounding_box']['extent']
    components = candidate['object_oriented_bounding_box']['orthogonal_offset_vectors']
    bbox = pv.Cube(
        center=(0, 0, 0),
        x_length=bb_extent[0],
        y_length=bb_extent[1],
        z_length=bb_extent[2],
    )

    #
    transformation = np.eye(4)
    transformation[:3, :3] = components
    transformation[:3, 3] = bb_center
    bbox.transform(transformation)
    return bbox


def create_bounding_boxes_pv(candidates):

    bboxes = []
    for candidate_pred in candidates:
        # create bounding box
        bbox = create_bounding_box_pv(candidate_pred)
        bboxes.append(bbox)

    return bboxes


def create_aneurysm_meshes(image):

    aneurysms = []
    for label in np.unique(image)[1:]:
        # create ground-truth aneurysm mesh
        mask_ixs = np.array(np.where(image == label)).T
        aneurysm = pv.PolyData(mask_ixs)
        aneurysm = aneurysm.delaunay_3d()
        aneurysms.append(aneurysm)

    return aneurysms


def rotate_aneurysm(aneurysm_np, bb_center, components, factor=1.):
    # rotate around bounding box center
    scaling = np.eye(4)
    scaling[:3, :3] *= 1/factor
    trans1 = np.eye(4)
    trans1[:3, 3] = -bb_center
    rot = np.eye(4)
    rot[:3, :3] = components
    trans2 = np.eye(4)
    trans2[:3, 3] = bb_center
    t = np.dot(trans2, np.dot(rot, np.dot(trans1, scaling)))

    aneurysm_nb = nib.Nifti1Image(aneurysm_np, affine=np.eye(4))
    return resample_img(aneurysm_nb, target_affine=t, interpolation='nearest')


def aneurysm_to_aligned_world_vectors(mask, affine, bb_center, components):

    # voxel to world coordinates
    voxel_coordinates = np.array(np.where(mask == 1)).T
    voxel_coordinates_ext = np.concatenate((voxel_coordinates, np.ones((voxel_coordinates.shape[0], 1))), axis=1)
    world_coordinates_ext = (affine @ voxel_coordinates_ext.T).T
    world_coordinates = world_coordinates_ext[:, :3]

    #
    return rotate_aneurysm_vectors(world_coordinates, bb_center, components)


def aneurysm_coverage(world_rot, bb_center, bb_extent):

    #
    bb_min = (bb_center - bb_extent / 2)
    bb_max = (bb_center + bb_extent / 2)

    #
    return world_rot[np.logical_and(
        np.all(world_rot[:,] >= bb_min - 1e-6, axis=1),
        np.all(world_rot[:,] <= bb_max + 1e-6, axis=1)
    )].shape[0]


def rotate_aneurysm_vectors(aneurysm_np, bb_center, components):

    #
    trans1 = np.eye(4)
    trans1[:3, 3] = -bb_center
    rot = np.eye(4)
    rot[:3, :3] = components
    trans2 = np.eye(4)
    trans2[:3, 3] = bb_center
    affine = np.dot(trans2, np.dot(rot, trans1))

    #
    aneurysm_np_ext = np.concatenate((aneurysm_np, np.ones((aneurysm_np.shape[0], 1))), axis=1)
    return (affine @ aneurysm_np_ext.T).T[:, :3]


def bounding_box_fit(world_rot, bb_center, bb_extent):

    #
    bb_min = (bb_center - bb_extent / 2)
    bb_max = (bb_center + bb_extent / 2)

    #
    mask_min = np.min(world_rot, axis=0)
    mask_max = np.max(world_rot, axis=0)

    #
    distances = np.concatenate((np.abs(mask_min - bb_min), np.abs(mask_max - bb_max)))
    return np.max(distances)


def calculate_inverse_affine(affine):
    inverse_affine = np.eye(4)
    inverse = np.linalg.inv(affine[:3, :3])

    #
    inverse_affine[:3, :3] = inverse
    inverse_affine[:3, 3] = -inverse @ affine[:3, 3]
    return inverse_affine


class CadaDetection(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=JSONLoader(),
            validators=(
                ExpectedColumnNamesValidator(expected=("dataset_id", "processing_time_in_seconds", "candidates")),
            ),
            join_key="dataset_id",
        )

    def score_case(self, *, idx, case) -> Dict:

        # load mask file
        case_id = case['dataset_id']
        file_path = Path('ground-truth-data/masks') / (case_id + '_labeledMasks.nii.gz')
        image = nib.load(file_path)
        image_np = image.get_fdata()

        #
        affine = image.affine
        inverse_affine = calculate_inverse_affine(affine)

        # find detections
        nr_candidates_gt, nr_candidates_pred = len(case['candidates_ground_truth']), len(case['candidates_prediction'])
        detection_matrix = np.zeros((nr_candidates_gt, nr_candidates_pred))
        for ix, candidate in enumerate(case['candidates_prediction']):

            #
            try:
                position = (inverse_affine @ np.array(candidate['position'] + [1])).astype(np.int)
                voxel = image_np[position[0], position[1], position[2]]
                if voxel > 0.:
                    detection_matrix[int(voxel - 1), ix] = 1.
            except Exception as e:
                pass

        # false-negatives, true-positives, false-negatives
        fp = np.sum(np.sum(detection_matrix, axis=0) == 0.)
        tp = np.sum(np.sum(detection_matrix, axis=1) > 0.)
        fn = np.sum(np.sum(detection_matrix, axis=1) == 0.)

        # calculate coverage of aneurysm and bounding box fit
        coverages, bbox_fits, total_volume, gt_labels = [], [], [], []
        for det, candidate in zip(detection_matrix.T, case['candidates_prediction']):

            #
            if 'object_oriented_bounding_box' not in candidate or np.sum(det) == 0.:
                continue

            #
            bb_center_world = np.array(candidate['position'])
            bbox = candidate['object_oriented_bounding_box']
            components = np.array(bbox['orthogonal_offset_vectors']).T
            bb_extent_world = np.array(bbox['extent'])

            #
            index_gt = np.where(det == 1.)[0] + 1
            aneurysm_np = (image_np == index_gt).astype(np.float)

            # convert nifti to axis aligned world vectors
            aligned_world_vectors = aneurysm_to_aligned_world_vectors(aneurysm_np, affine, bb_center_world, components)

            #
            coverage = aneurysm_coverage(aligned_world_vectors, bb_center_world, bb_extent_world)
            bbox_fit = bounding_box_fit(aligned_world_vectors, bb_center_world, bb_extent_world)
            coverages.append(float(coverage / aligned_world_vectors.shape[0]))
            bbox_fits.append(float(bbox_fit))
            total_volume.append(float(aligned_world_vectors.shape[0]))
            gt_labels.append(float(index_gt))

        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'coverage': coverages,
            'bbox_fit': bbox_fits,
            'total_volume': total_volume,
            'GT_labels': gt_labels
        }

    def score_aggregates(self):

        #
        tp = np.sum(self._case_results['tp'])
        fp = np.sum(self._case_results['fp'])
        fn = np.sum(self._case_results['fn'])

        #
        coverages = np.concatenate(self._case_results['coverage'].apply(np.array))
        bbox_fits = np.concatenate(self._case_results['bbox_fit'].apply(np.array))
        total_volume = np.concatenate(self._case_results['total_volume'].apply(np.array))
        volume_weights = total_volume / np.sum(total_volume)

        # recall/precision
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f_beta = 5 * (precision * recall) / (4 * precision + recall)

        return {
            'Recall': recall,
            'Precision': precision,
            'F2_score': f_beta,
            'CoverageSum': np.sum(coverages),
            'CoverageMean': np.mean(coverages),
            'CoverageWeightedMean': np.sum(volume_weights * coverages),
            'BBoxFitSum': np.sum(bbox_fits),
            'BBoxFitMean': np.mean(bbox_fits),
            # 'Total_volume': np.sum(total_volume)
        }

    def save(self):
        metrics = dict()
        metrics['aggregates'] = self._metrics['aggregates']
        with open(self._output_file, "w") as f:
            f.write(json.dumps(metrics))


if __name__ == "__main__":
    CadaDetection().evaluate()
