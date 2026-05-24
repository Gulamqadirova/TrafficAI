from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import config


class CentroidTracker:
    """Greedy nearest-neighbour centroid tracker for a single class."""

    def __init__(self, max_distance: int = config.MAX_TRACK_DISTANCE,
                 max_disappeared: int = config.MAX_DISAPPEARED) -> None:
        self.next_id = 0
        self.objects: "OrderedDict[int, Tuple[int,int]]" = OrderedDict()
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def _register(self, c): self.objects[self.next_id] = c; self.disappeared[self.next_id] = 0; self.next_id += 1
    def _deregister(self, oid): del self.objects[oid]; del self.disappeared[oid]

    def update(self, centroids: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        pts = np.array(centroids, float)
        if len(self.objects) == 0:
            for c in pts:
                self._register((int(c[0]), int(c[1])))
            return dict(self.objects)

        ids = list(self.objects.keys())
        obj = np.array(list(self.objects.values()), float)
        d = np.linalg.norm(obj[:, None, :] - pts[None, :, :], axis=2)
        rows = d.min(axis=1).argsort()
        cols = d.argmin(axis=1)[rows]
        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            if r in used_r or c in used_c or d[r, c] > self.max_distance:
                continue
            oid = ids[r]
            self.objects[oid] = (int(pts[c][0]), int(pts[c][1]))
            self.disappeared[oid] = 0
            used_r.add(r); used_c.add(c)
        for r in set(range(d.shape[0])) - used_r:
            oid = ids[r]; self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)
        for c in set(range(d.shape[1])) - used_c:
            self._register((int(pts[c][0]), int(pts[c][1])))
        return dict(self.objects)


class LineCounter:
    """Counts crossings of a vertical line, by direction, for one class."""

    def __init__(self, line_x: int) -> None:
        self.line_x = line_x
        self.prev_x: Dict[int, int] = {}
        self.left_to_right = 0
        self.right_to_left = 0

    def update(self, objects: Dict[int, Tuple[int, int]]) -> None:
        for oid, (cx, _cy) in objects.items():
            if oid in self.prev_x:
                px = self.prev_x[oid]
                if px < self.line_x <= cx:
                    self.left_to_right += 1
                elif px >= self.line_x > cx:
                    self.right_to_left += 1
            self.prev_x[oid] = cx

    @property
    def total(self) -> int:
        return self.left_to_right + self.right_to_left


class MultiClassTracker:
    """Holds an independent tracker + counter for each monitored class."""

    def __init__(self, line_x: int) -> None:
        self.trackers = {cid: CentroidTracker() for cid in config.TARGET_CLASS_IDS}
        self.counters = {cid: LineCounter(line_x) for cid in config.TARGET_CLASS_IDS}

    def update(self, detections, centroid_fn):
        per_class_objs = {}
        for cid in config.TARGET_CLASS_IDS:
            cents = [centroid_fn(d["bbox"]) for d in detections if d["class_id"] == cid]
            objs = self.trackers[cid].update(cents)
            self.counters[cid].update(objs)
            per_class_objs[cid] = objs
        return per_class_objs


if __name__ == "__main__":
    t = MultiClassTracker(line_x=100)
    print("Trackers for classes:", list(t.trackers.keys()))
