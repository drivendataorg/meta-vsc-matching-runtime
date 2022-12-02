from collections import defaultdict
import enum
from itertools import groupby
import json
from math import sqrt
from pathlib import Path
from typing import Collection, Dict, List, NamedTuple, Optional, Tuple
import pandas as pd
import typer


class Axis(enum.Enum):
    QUERY = enum.auto()
    REF = enum.auto()


class Match(NamedTuple):
    """A ground-truth match or predicted match."""

    query_id: str
    reference_id: str
    query_start: float
    query_end: float
    reference_start: float
    reference_end: float
    score: float = 1.0

    @property
    def pair_id(self):
        return self.query_id, self.reference_id

    def interval(self, axis: Axis) -> Tuple[float, float]:
        if axis == Axis.QUERY:
            return self.query_start, self.query_end
        else:
            return self.reference_start, self.reference_end

    def intersection_area(self, other: "Match") -> float:
        # Compared pairs should be identical
        if self.pair_id != other.pair_id:
            raise ValueError("Calculating intersection between incompatible matches.")
        # Compute the intersection boarders
        inter_q_start = max(self.query_start, other.query_start)
        inter_r_start = max(self.reference_start, other.reference_start)
        inter_q_end = min(self.query_end, other.query_end)
        inter_r_end = min(self.reference_end, other.reference_end)

        # Compute the area of intersection rectangle
        return abs(
            max((inter_q_end - inter_q_start, 0))
            * max((inter_r_end - inter_r_start), 0)
        )

    def overlaps(self, other: "Match") -> bool:
        return self.intersection_area(other) > 0.0


class Intervals:
    # Non-overlapping, ordered by interval start.
    intervals: List[Tuple[float, float]]

    def __init__(self, intervals: Optional[List[Tuple[float, float]]] = None):
        self.intervals = intervals or []
        self._dedup()

    def add(self, interval: Tuple[float, float]):
        """Add an interval."""
        self.intervals.append(interval)
        self._dedup()

    def union(self, intervals: "Intervals") -> "Intervals":
        return Intervals(self.intervals + intervals.intervals)

    def total_length(self):
        return sum(end - start for start, end in self.intervals)

    def intersect_length(self, intervals: "Intervals") -> "Intervals":
        """Compute the total_length of the intersection of two Intervals.

        This works by taking the sum of their lengths, and subtracting
        the length of their union.

        |A n B| = |A| + |B| - |A U B|
        """
        union = self.union(intervals)
        return self.total_length() + intervals.total_length() - union.total_length()

    def _dedup(self):
        if len(self.intervals) <= 1:
            return
        deduped = []
        intervals = sorted(self.intervals)
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                # Overlap case
                current_end = max(end, current_end)
            else:
                # Non-overlap case
                deduped.append((current_start, current_end))
                current_start, current_end = start, end
        deduped.append((current_start, current_end))
        self.intervals = deduped

    def __str__(self):
        return str(self.intervals)

    __repr__ = __str__


class VideoPairEvaluator:
    """An object that calculates intersections between predicted and ground truth
    matches that belong to one video.

    Provide functionalities for the combination of new predictions with the
    existing ones and the computation of their intersection with the gt bboxes,
    ignoring the gt bboxes that do not overlap with any prediction.
    """

    gts: List[Match]
    preds: List[Match]

    def __init__(
        self,
    ):
        self.intersections = {axis: 0 for axis in Axis}
        self.totals = {axis: 0 for axis in Axis}
        self.gts = []
        self.preds = []

    def total_gt_length(self, axis: Axis) -> int:
        return Intervals([gt.interval(axis) for gt in self.gts]).total_length()

    def total_pred_length(self, axis: Axis) -> int:
        return Intervals([pred.interval(axis) for pred in self.preds]).total_length()

    def gt_overlaps(self, gt: Match) -> bool:
        """Checks if the provided gt bbox overlaps with at least one pred bbox."""
        for pred in self.preds:
            if gt.overlaps(pred):
                return True
        return False

    def add_gt(self, bbox: Match):
        self.gts.append(bbox)

    def add_prediction(self, bbox: Match) -> Tuple[Dict, Dict]:
        """Add a prediction to the corresponding list and calculates the
        differences in the intersections with the gt and the total video
        length covered for both query and reference axes.
        """
        self.preds.append(bbox)
        # A subset of GTs to consider for intersection (but not total GT length).
        gts_to_consider = [gt for gt in self.gts if self.gt_overlaps(gt)]

        intersect_deltas = {}
        total_deltas = {}

        for axis in Axis:
            pred_ints = Intervals([pred.interval(axis) for pred in self.preds])
            gt_ints = Intervals([gt.interval(axis) for gt in gts_to_consider])
            # New intersection and total length on this axis
            intersect_length = pred_ints.intersect_length(gt_ints)
            prediction_length = pred_ints.total_length()
            # Compute differences
            intersect_deltas[axis] = intersect_length - self.intersections[axis]
            total_deltas[axis] = prediction_length - self.totals[axis]
            # Update with new values
            self.intersections[axis] = intersect_length
            self.totals[axis] = prediction_length

        return intersect_deltas, total_deltas


def match_metric(gts: Collection[Match], predictions: Collection[Match]):
    r"""V2 metric:

    Computes the AP based on the VCSL approach for the
    calculation of Precision and Recall.

    AP = \sum_{i=1}^N P(i) ΔR(i)

    where, P(i) = sqrt(P_q * P_r) and R(i) = sqrt(R_q * R_r)
    calculated as in the VCSL.
    """

    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

    # Initialize video pairs and load their gt bboxs
    video_pairs = defaultdict(VideoPairEvaluator)
    for gt in gts:
        video_pairs[gt.pair_id].add_gt(gt)

    # Get the total gt length for each axis
    gt_total_lengths = {axis: 0 for axis in Axis}
    for _, v in video_pairs.items():
        for axis in Axis:
            gt_total_lengths[axis] += v.total_gt_length(axis)

    # Loop through the predictions
    recall = 0.0
    metric = 0.0
    intersections = {axis: 0 for axis in Axis}
    totals = {axis: 0 for axis in Axis}

    # Group predictions by score to break ties consistently
    for _, preds in groupby(predictions, key=lambda x: x.score):
        # Update precision and recall within a given group before updating metric
        for pred in preds:
            pair_id = pred.pair_id
            # Given a new prediction, we only need the differences in the intersection with
            # gt and total video length covered for both query and reference axes.
            intersection_deltas, total_deltas = video_pairs[pair_id].add_prediction(
                pred
            )

            recalls = {}
            precisions = {}
            for axis in Axis:
                # Accumulate the differences to the corresponding values
                intersections[axis] += intersection_deltas[axis]
                totals[axis] += total_deltas[axis]

        for axis in Axis:
            recalls[axis] = intersections[axis] / gt_total_lengths[axis]
            precisions[axis] = intersections[axis] / totals[axis]

        new_recall = sqrt(recalls[Axis.QUERY] * recalls[Axis.REF])
        precision = sqrt(precisions[Axis.QUERY] * precisions[Axis.REF])

        # Compute metric
        delta_recall = new_recall - recall
        metric += precision * delta_recall
        recall = new_recall

    return metric


def main(
    submission_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to submission file (submission.csv)",
    ),
    ground_truth_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to ground truth CSV file.",
    ),
):
    """Evaluate a submission for the Meta AI Video Similarity Challenge, Matching Track."""
    predicted = pd.read_csv(submission_path)
    actual = pd.read_csv(ground_truth_path)

    pred = [Match(**tup._asdict()) for tup in predicted.itertuples(index=False)]
    gt = [Match(**tup._asdict()) for tup in actual.dropna().itertuples(index=False)]
    metric = match_metric(gt, pred)

    typer.echo(
        json.dumps(
            {
                "mean_average_precision": metric,
            }
        )
    )


if __name__ == "__main__":
    typer.run(main)
