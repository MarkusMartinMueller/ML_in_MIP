import argparse
from pathlib import Path
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Converting CSV to JSON format.')
    parser.add_argument('-file', metavar='-f', type=str)

    return parser.parse_args()


def main():

    # get arguments
    args = parse_args()
    file = Path(args.file)

    # load CSV
    df = pd.read_csv(file, sep=';')
    df.columns = ['dataset_id', 'rupture_status']
    df['processing_time_in_seconds'] = 42

    # prepare json
    result = {
        'grand_challenge_username': 'ICM',
        "used_hardware_specification": {
            "CPU": "Intel Core i9 9900K 8x 3.60GHz",
            "GPU": "NVIDIA RTX 2080 Ti",
            "#GPUs": 1,
            "RAM_in_GB": 4,
            "additional_remarks": "special hardware requirements, other comments"
        },
        'task_3_results': df.to_dict('records')
    }

    # write to file
    with open(str(file.parent / 'reference.json'), 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
