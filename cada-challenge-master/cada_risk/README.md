# Aneurysm rupture risk evaluation (task 3)

## Data

To run the evaluation on the training data locally, copy
`/training-data/task3/reference.json` to the `ground-truth` and the `test` directories.

Evaluating the training data on your own predictions, replace the file in the `test` 
directory by your own `.json` in the [defined format](https://cada-rre.grand-challenge.org/Submission-Details/).

## Run

To test the evaluation docker needs to be running on your machine.

Run the evaluation with `./test.sh`.