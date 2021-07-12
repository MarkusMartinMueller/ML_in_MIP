import json
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

from evalutils import ClassificationEvaluation
from evalutils.io import FileLoader
from evalutils.validators import ExpectedColumnNamesValidator


class JSONLoader(FileLoader):
    def __init__(self):
        super().__init__()

    def load(self, fname):

        if fname.suffix != '.json':
            return None

        with open(fname) as json_file:
            data = json.load(json_file)
            # TODO: validate json
            return pd.DataFrame(data['task_3_results'])


class CadaRisk(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=JSONLoader(),
            validators=(ExpectedColumnNamesValidator(expected=("dataset_id", "rupture_status")),),
            join_key="dataset_id",
        )

    def score_aggregates(self):

        accuracy = accuracy_score(self._cases["rupture_status_ground_truth"], self._cases["rupture_status_prediction"])
        precision = precision_score(self._cases["rupture_status_ground_truth"], self._cases["rupture_status_prediction"])
        recall = recall_score(self._cases["rupture_status_ground_truth"], self._cases["rupture_status_prediction"])
        f1 = f1_score(self._cases["rupture_status_ground_truth"], self._cases["rupture_status_prediction"])
        f_beta = fbeta_score(self._cases["rupture_status_ground_truth"], self._cases["rupture_status_prediction"], beta=2)

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "F2_score": f_beta
        }

    def save(self):
        metrics = dict()
        metrics['aggregates'] = self._metrics['aggregates']
        with open(self._output_file, "w") as f:
            f.write(json.dumps(metrics))


if __name__ == "__main__":
    CadaRisk().evaluate()
