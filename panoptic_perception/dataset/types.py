import numpy as np

from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class Bbox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def valid_bbox(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    def xyxy2xywh(self) -> Tuple[float, float, float, float]:
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        cx = self.x1 + w / 2
        cy = self.y1 + h / 2
        return cx, cy, w, h

    def to_list(self) -> List[float]:
        """Pixel xyxy as a list: [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: List[float]) -> "Bbox":
        """Reverse of to_list. Expects [x1, y1, x2, y2] in pixels."""
        x1, y1, x2, y2 = coords
        return cls(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


@dataclass
class ObjDetInstance:
    bbox: Bbox
    label_id: int
    label: str = None


@dataclass
class LanePoly:
    points: np.array
    category: str


@dataclass
class FrameLaneDetections:
    lane_polys: List[LanePoly] = field(default_factory=list)

    def __iter__(self):
        return iter(self.lane_polys)

    def __len__(self):
        return len(self.lane_polys)


@dataclass
class FrameObjDetections:
    detections: List[ObjDetInstance] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)


@dataclass
class FrameData:
    image: np.array
    image_path: str

    seg: Optional[np.ndarray] = None
    drivable: Optional[np.ndarray] = None

    frame_detections: Optional[FrameObjDetections] = None
    lane_polys: Optional[FrameLaneDetections] = None

    def labels_array(self) -> np.ndarray:
        """Detections as (N, 5) [cls, cx, cy, w, h] normalized to image size.

        Returns (0, 5) zeros if no detections. Image dimensions are read from
        self.image.shape — image must be HWC.
        """
        if self.frame_detections is None or len(self.frame_detections) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        h, w = self.image.shape[:2]
        rows = []
        for det in self.frame_detections.detections:
            cx, cy, bw, bh = det.bbox.xyxy2xywh()
            rows.append([det.label_id, cx / w, cy / h, bw / w, bh / h])
        return np.array(rows, dtype=np.float32)

    def set_labels_array(self, labels: np.ndarray) -> None:
        """Inverse of labels_array. Writes back into self.frame_detections,
        scaling normalized coords to pixels using current self.image size.
        Preserves existing FrameObjDetections.attributes if any.
        """
        attributes = (
            self.frame_detections.attributes
            if self.frame_detections is not None else {}
        )
        if labels is None or len(labels) == 0:
            self.frame_detections = FrameObjDetections(
                detections=[], attributes=attributes
            )
            return

        h, w = self.image.shape[:2]
        dets: List[ObjDetInstance] = []
        for row in labels:
            cls, cx, cy, bw, bh = row
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            dets.append(ObjDetInstance(
                bbox=Bbox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                label_id=int(cls),
            ))
        self.frame_detections = FrameObjDetections(
            detections=dets, attributes=attributes
        )

    def lane_polys_legacy(self) -> Optional[List[dict]]:
        """LanePolys as the legacy list-of-dict format used by augmentations.
        Returns None if no lane_polys to keep "lane_polys is None" gating
        unchanged downstream.
        """
        if self.lane_polys is None:
            return None
        return [
            {"points": lp.points, "category": lp.category}
            for lp in self.lane_polys.lane_polys
        ]

    def set_lane_polys_legacy(self, polys: Optional[List[dict]]) -> None:
        """Inverse of lane_polys_legacy."""
        if polys is None:
            self.lane_polys = None
            return
        self.lane_polys = FrameLaneDetections(
            lane_polys=[
                LanePoly(points=p["points"], category=p["category"])
                for p in polys
            ]
        )


class DatasetMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"
