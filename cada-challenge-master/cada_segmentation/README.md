# Aneurysm segmentation evaluation (task 2)

## Data

To run the evaluation on the training data locally, copy the `*labeledMasks.nii.gz` 
NIFTI files to the `ground-truth` and `test` directories.

Evaluating the training data on your own predictions, replace the files in the `test` 
directory by your own `.nii.gz` NIFTI files.

## Run

To test the evaluation docker needs to be running on your machine.

Run the evaluation with `./test.sh`.