import json
import numpy as np
import pandas as pd
import nibabel as nib
from evalutils.exceptions import FileLoaderError
from scipy.ndimage import center_of_mass
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import jaccard_score

from evalutils.stats import hausdorff_distance, mean_contour_distance
from evalutils import ClassificationEvaluation
from evalutils.io import ImageLoader
from evalutils.validators import UniqueImagesValidator, UniquePathIndicesValidator

from typing import Dict


class NiftiLoader(ImageLoader):

    def __init__(self):
        super().__init__()

    @staticmethod
    def load_image(fname):

        #
        if fname.suffix != '.gz' and fname.suffix != '.nii':
            raise FileLoaderError('Could not load {}'.format(str(fname)))

        # load file
        mask = nib.load(str(fname))
        return mask

    @staticmethod
    def hash_image(image):
        if image is not None:
            return int(hash(image.get_fdata().tostring()))


class CadaSegmentation(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=NiftiLoader(),
            validators=(),
                # UniquePathIndicesValidator(),
                # UniqueImagesValidator(),
            #),
            file_sorter_key=lambda fname: fname.stem.split('.')[0]
        )

    def score_case(self, *, idx, case) -> Dict:
        gt_path = case["path_ground_truth"]
        pred_path = case["path_prediction"]

        # Load the images for this case
        gt_nifti = self._file_loader.load_image(gt_path)
        pred_nifti = self._file_loader.load_image(pred_path)
        gt_voxelspacing, pred_voxelspacing = gt_nifti.header['pixdim'][1:4], pred_nifti.header['pixdim'][1:4]
        volume_per_voxel = np.prod(gt_voxelspacing)
        gt, pred = gt_nifti.get_fdata(), pred_nifti.get_fdata()

        #
        gt_labels = np.unique(gt)[1:]
        pred_labels = np.unique(pred)[1:]

        # check if structures are labeled correctly
        # expected increasing labels starting at 1.
        if not np.array_equal(pred_labels, np.arange(1, len(pred_labels) + 1)):
            raise ValueError('Aneurysms are not labeled correctly ({}). The structures are expected to be increasing '
                             'by one starting at 1'.format(pred_path))

        # voxel spacing in ground-truth and prediction needs to be equal
        if not np.array_equal(gt_voxelspacing, pred_voxelspacing):
            raise ValueError('Voxel spacing not equal in ground-truth and prediction ({}).'.format(pred_path))

        # calculate center of mass for each structure
        gt_com = center_of_mass(gt, labels=gt, index=gt_labels)
        pred_com = center_of_mass(pred, labels=pred, index=pred_labels)
        gt_com, pred_com = np.array(gt_com).reshape(-1, 3), np.array(pred_com).reshape(-1, 3)

        if gt_com.shape[0] < 1 or gt_com.shape[0] < 1:
            print('No structure in "{}", mask-sum = {}'.format(pred_path, np.sum(pred)))
            return None

        # jaccard
        jaccard = jaccard_score((gt > 0.).flatten(), (pred > 0.).flatten())

        #
        gt_volume = []
        for gt_ix in gt_labels:
            gt_aneurysm = gt == gt_ix

            # calculate volume of structure
            gt_volume.append(float(np.sum(gt_aneurysm) * volume_per_voxel))

        #
        if np.sum(pred_com) == 0.:
            return {
                'Jaccard': jaccard,
                'HausdorffDistance': [1000. for _ in gt_volume],
                'MeanDistance': [1000. for _ in gt_volume],
                # 'CenterOfMassGt': gt_com,
                # 'CenterOfMassPred': pred_com,
                'VolumeGt': gt_volume,
                'VolumePred': [0. for _ in gt_volume],
                'pred_fname': pred_path.name,
                'gt_fname': gt_path.name,
            }

        # calculate correspondence
        gt_pred_correspondence = pairwise_distances_argmin(gt_com, pred_com)

        hausdorff, mean_distance, pred_volume = [], [], []
        for gt_ix, corr in zip(gt_labels, gt_pred_correspondence + 1):
            # current aneurysm
            gt_aneurysm = gt == gt_ix
            pred_aneurysm = pred == corr

            # calculate hausdorff on structures
            hausdorff.append(float(hausdorff_distance(gt_aneurysm, pred_aneurysm, voxelspacing=gt_voxelspacing)))

            # calculate mean contour distance on structures
            mean_distance.append(float(mean_contour_distance(gt_aneurysm, pred_aneurysm, voxelspacing=gt_voxelspacing)))

            # calculate volume of structure
            pred_volume.append(float(np.sum(pred_aneurysm) * volume_per_voxel))

        return {
            'Jaccard': jaccard,
            'HausdorffDistance': hausdorff,
            'MeanDistance': mean_distance,
            # 'CenterOfMassGt': gt_com,
            # 'CenterOfMassPred': pred_com,
            'VolumeGt': gt_volume,
            'VolumePred': pred_volume,
            'pred_fname': pred_path.name,
            'gt_fname': gt_path.name,
        }

    def score_aggregates(self) -> Dict:
        aggregate_results = {}

        # pearson correlation coefficient r
        gt_volume = np.concatenate(self._case_results['VolumeGt'].apply(np.array))
        pred_volume = np.concatenate(self._case_results['VolumePred'].apply(np.array))
        aggregate_results['VolumePearsonR'] = pearsonr(gt_volume, pred_volume)[0]

        # absolute volume difference bias + std.
        diff = np.abs(gt_volume - pred_volume)
        aggregate_results['VolumeBias'] = np.mean(diff)
        aggregate_results['VolumeStd'] = np.std(diff)

        # Hausdorff distance aggregate
        hausdorff = np.concatenate(self._case_results['HausdorffDistance'].apply(np.array))
        aggregate_results['HausdorffDistance'] = self.aggregate_series(series=pd.Series(hausdorff))

        # Mean contour distance aggregate
        mean_distance = np.concatenate(self._case_results['MeanDistance'].apply(np.array))
        aggregate_results['MeanDistance'] = self.aggregate_series(series=pd.Series(mean_distance))

        for col in self._case_results.columns:
            if col in ['VolumeGt', 'VolumePred', 'HausdorffDistance', 'MeanDistance']:
                continue

            aggregate_results[col] = self.aggregate_series(
                series=self._case_results[col]
            )

        return aggregate_results

    def save(self):
        metrics = dict()
        metrics['aggregates'] = self._metrics['aggregates']
        with open(self._output_file, "w") as f:
            f.write(json.dumps(metrics))


if __name__ == "__main__":
    CadaSegmentation().evaluate()
