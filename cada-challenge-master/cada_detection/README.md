# Aneurysm detection evaluation (task 1)

## Data

To run the evaluation on the training data locally, copy
`/training-data/task1/reference.json` to the `ground-truth` and the `test` directories.

Furthermore, copy the `*labeledMasks.nii.gz` NIFTI files to the `ground-truth-data/masks` directory.

Evaluating the training data on your own predictions, replace the file in the `test` 
directory by your own `.json` in the [defined format](https://cada.grand-challenge.org/Submission-Details/).

## Run

To test the evaluation docker needs to be running on your machine.

Run the evaluation with `./test.sh`.