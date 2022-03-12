"""
This script should be used for the evaluation of your solution. To use it you need to install ir_measures
with `pip install ir_measures`

Usage:
python evaluation.py predictions.json gold.json

The prediction.json file should contain the retrieved job ids alongside their relevance scores for each job-query. E.g.:
{
    "100": {"10": 3., "20": 4.},
    "200": {"5": 10. },
    ...
}
Where "100", "200" are the ids of the jobs used as queries. "10", "20", "5" are the ids of the retrieved jobs.
3., 4., 10. are the relevance scores.

The gold.json file should contain the relevant job ids for each job-query. E.g.:
{
    "100": {"10": 1, "25": 1},
    "200": {"5": 1 },
    ...
}
Where "100", "200" are the ids of jobs used as queries. "10", "25", "5" are the ids of the relevant jobs.
"""

import json
import ir_measures
import os.path
import sys

from ir_measures import AP, Rprec


if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
    print("Usage:")
    print("python evaluation.py predictions.json gold.json")
    exit(-1)

with open(sys.argv[1]) as fr:
    predictions = json.load(fr)

with open(sys.argv[2]) as fr:
    gold = json.load(fr)

print(ir_measures.calc_aggregate([AP, Rprec], gold, predictions))
