"""
synthetic_data.py
=================
Synthetic video-frame generator for benchmarking and unit testing the
Color-Based Object Detection System without requiring a physical camera.

The generator renders configurable scenes with:
- Moving coloured geometric shapes (circles, rectangles, triangles)
- Per-frame Gaussian sensor noise (sigma = 0 to 25 DN)
- Simulated ambient brightness scaling (Lux -> gain factor)
- Specular glare patches (bright white ellipses)
- Partial occlusion rectangles (dark grey occluder overlay)

Ground-Truth Output
-------------------
For every generated frame the module also returns a list of
GroundTruthObject named-tuples that contain the exact bounding box
and colour name of each drawn shape, enabling IoU-based metric computation
in evaluate.py.

Usage
-----
    from benchmarks.synthetic_data import SyntheticSceneGenerator, LuxLevel

    gen = SyntheticSceneGenerator(width=640, height=480, seed=42)
    for frame, gt_objects in gen.stream(num_frames=200, lux=LuxLevel.NORMAL):
        ...
"""

import math
import random
from collections import namedtuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases and named-tuples
# ---------------------------------------------------------------------------

GroundTruthObject = namedtuple("GroundTruthObject", ["color_name", "bbox", "shape"])
"""
Ground truth record for a single rendered object.

Fields
------
color_name : str
    One of "Red", "Green", "Blue", "Yellow".
bbox : tuple
    Axis-aligned bounding box as (x, y, w, h) in pixels.
shape : str
    One of "circle", "rectangle", "triangle".
"""


# ---------------------------------------------------------------------------
# Illuminance simulation
# ---------------------------------------------------------------------------

class LuxLevel(object):
    """Discrete illuminance presets (Lux).

    Attributes
    ----------
    VERY_DIM : int
        Moonlit room (~50 lux)
    DIM : int
        Corridor / night-time office (~200 lux)
    NORMAL : int
        Well-lit office (~400 lux)
    BRIGHT : int
        Retail store / studio (~800 lux)
    VERY_BRIGHT : int
        Production floor (~1200 lux)
    OUTDOOR : int
        Overcast daylight (~1600 lux)
    """

    VERY_DIM    = 50
    DIM         = 200
    NORMAL      = 400
    BRIGHT      = 800
    VERY_BRIGHT = 1200
    OUTDOOR     = 1600


def lux_to_gain(lux, reference_lux=None):
    """Convert an illuminance value to a linear brightness gain factor.

    Parameters
    ----------
    lux : int
        Target illuminance in Lux.
    reference_lux : int, optional
        Illuminance at which gain equals 1.0 (neutral). Default: NORMAL.

    Returns
    -------
    float
        Brightness gain factor clamped to [0.1, 3.0].
    """
    if reference_lux is None:
        reference_lux = LuxLevel.NORMAL
    gain = lux / max(reference_lux, 1)
    return float(np.clip(gain, 0.1, 3.0))


# ---------------------------------------------------------------------------
# Object motion model (plain object for Python 3.6 compatibility)
# ---------------------------------------------------------------------------

class _MovingObject(object):
    """Internal state for a single animating shape."""

    def __init__(self, color_name, bgr, shape, cx, cy, radius, vx, vy,
                 angle=0.0, spin=0.0):
        self.color_name = color_name
        self.bgr        = bgr
        self.shape      = shape
        self.cx         = cx
        self.cy         = cy
        self.radius     = radius
        self.vx         = vx
        self.vy         = vy
        self.angle      = angle
        self.spin       = spin


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticSceneGenerator(object):
    """Generates synthetic BGR video frames with annotated ground truth.

    Parameters
    ----------
    width, height : int
        Frame dimensions in pixels.
    num_objects : int
        Number of randomly spawned coloured shapes per scene.
    noise_sigma : float
        Standard deviation of additive Gaussian noise (DN).
    glare_probability : float
        Probability [0, 1] of adding a specular glare patch each frame.
    occlusion_probability : float
        Probability [0, 1] of adding a partial occlusion each frame.
    seed : int or None
        Random seed for reproducible sequences.

    Example
    -------
    >>> gen = SyntheticSceneGenerator(640, 480, seed=0)
    >>> frames_gt = list(gen.stream(100, lux=LuxLevel.NORMAL))
    >>> frame, gt = frames_gt[0]
    >>> print(gt[0].color_name, gt[0].bbox)
    """

    _COLOR_CATALOGUE = {
        "Red":    (0,   30,  220),
        "Green":  (30,  200,  30),
        "Blue":   (200,  50,  30),
        "Yellow": (0,   220, 220),
    }

    _SHAPES = ["circle", "rectangle", "triangle"]

    def __init__(
        self,
        width=640,
        height=480,
        num_objects=4,
        noise_sigma=8.0,
        glare_probability=0.15,
        occlusion_probability=0.10,
        seed=42,
    ):
        self.width       = width
        self.height      = height
        self.num_objects = num_objects
        self.noise_sigma = noise_sigma
        self.glare_prob  = glare_probability
        self.occlusion_prob = occlusion_probability

        self._rng    = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        self._objects = []
        self._spawn_objects()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(self, num_frames, lux=None):
        """Yield (frame, ground_truth_list) for num_frames frames.

        Parameters
        ----------
        num_frames : int
            Total number of frames to generate.
        lux : int, optional
            Target illuminance used to scale brightness. Default: NORMAL.

        Yields
        ------
        frame : np.ndarray
            BGR uint8 numpy array of shape (height, width, 3).
        gt_list : list of GroundTruthObject
        """
        if lux is None:
            lux = LuxLevel.NORMAL
        gain = lux_to_gain(lux)

        for _ in range(num_frames):
            frame, gt_list = self._render_frame(gain)
            yield frame, gt_list
            self._step_physics()

    def generate_single(self, lux=None):
        """Generate and return a single frame without advancing the simulation."""
        if lux is None:
            lux = LuxLevel.NORMAL
        gain = lux_to_gain(lux)
        return self._render_frame(gain)

    def reset(self):
        """Re-spawn all objects at random starting positions."""
        self._objects = []
        self._spawn_objects()

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _spawn_objects(self):
        color_names = list(self._COLOR_CATALOGUE.keys())
        for i in range(self.num_objects):
            name   = color_names[i % len(color_names)]
            bgr    = self._COLOR_CATALOGUE[name]
            shape  = self._rng.choice(self._SHAPES)
            radius = self._rng.randint(20, 55)

            obj = _MovingObject(
                color_name=name,
                bgr=bgr,
                shape=shape,
                cx=float(self._rng.randint(radius, self.width  - radius)),
                cy=float(self._rng.randint(radius, self.height - radius)),
                radius=radius,
                vx=self._rng.uniform(-4.0, 4.0),
                vy=self._rng.uniform(-4.0, 4.0),
                spin=self._rng.uniform(-3.0, 3.0),
            )
            self._objects.append(obj)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _step_physics(self):
        for obj in self._objects:
            obj.cx    += obj.vx
            obj.cy    += obj.vy
            obj.angle  = (obj.angle + obj.spin) % 360.0

            if obj.cx - obj.radius < 0 or obj.cx + obj.radius > self.width:
                obj.vx = -obj.vx
                obj.cx = float(np.clip(obj.cx, obj.radius, self.width  - obj.radius))
            if obj.cy - obj.radius < 0 or obj.cy + obj.radius > self.height:
                obj.vy = -obj.vy
                obj.cy = float(np.clip(obj.cy, obj.radius, self.height - obj.radius))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self, gain):
        # 1. Background
        canvas = np.full((self.height, self.width, 3), 40, dtype=np.float32)

        # 2. Shapes
        gt_list = []
        for obj in self._objects:
            bbox = self._draw_shape(canvas, obj)
            gt_list.append(
                GroundTruthObject(
                    color_name=obj.color_name,
                    bbox=bbox,
                    shape=obj.shape,
                )
            )

        # 3. Brightness gain
        canvas *= gain

        # 4. Gaussian noise
        if self.noise_sigma > 0:
            noise   = self._np_rng.normal(0, self.noise_sigma, canvas.shape)
            canvas += noise

        # 5. Specular glare
        if self._rng.random() < self.glare_prob:
            self._add_glare(canvas)

        # 6. Partial occlusion
        if self._rng.random() < self.occlusion_prob:
            self._add_occlusion(canvas)

        # 7. Clip and cast
        frame = np.clip(canvas, 0, 255).astype(np.uint8)
        return frame, gt_list

    def _draw_shape(self, canvas, obj):
        cx, cy = int(obj.cx), int(obj.cy)
        r      = obj.radius
        bgr_f  = tuple(float(c) for c in obj.bgr)

        if obj.shape == "circle":
            cv2.circle(canvas, (cx, cy), r, bgr_f, -1, cv2.LINE_AA)
            x, y, w, h = cx - r, cy - r, 2 * r, 2 * r

        elif obj.shape == "rectangle":
            x, y, w, h = cx - r, cy - r, 2 * r, 2 * r
            cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr_f, -1)

        else:  # triangle
            pts = self._triangle_points(cx, cy, r, obj.angle)
            cv2.fillPoly(canvas, [pts], bgr_f)
            x, y, w, h = cv2.boundingRect(pts)

        x = max(0, x)
        y = max(0, y)
        w = min(w, self.width  - x)
        h = min(h, self.height - y)
        return (x, y, w, h)

    @staticmethod
    def _triangle_points(cx, cy, r, angle_deg):
        angles = [angle_deg + 90, angle_deg + 210, angle_deg + 330]
        pts = np.array(
            [
                [
                    int(cx + r * math.cos(math.radians(a))),
                    int(cy - r * math.sin(math.radians(a))),
                ]
                for a in angles
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        return pts

    def _add_glare(self, canvas):
        gx        = self._rng.randint(0, self.width)
        gy        = self._rng.randint(0, self.height)
        gw        = self._rng.randint(20, 80)
        gh        = self._rng.randint(15, 50)
        intensity = self._rng.uniform(180, 255)
        cv2.ellipse(
            canvas, (gx, gy), (gw, gh),
            self._rng.randint(0, 180), 0, 360,
            (intensity, intensity, intensity), -1,
        )

    def _add_occlusion(self, canvas):
        ox    = self._rng.randint(0, self.width  // 2)
        oy    = self._rng.randint(0, self.height // 2)
        ow    = self._rng.randint(40, self.width  // 3)
        oh    = self._rng.randint(40, self.height // 3)
        shade = self._rng.randint(20, 60)
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), (shade, shade, shade), -1)
