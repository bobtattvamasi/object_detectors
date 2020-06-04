# vim: expandtab:ts=4:sw=4

import math
from collections import deque


def is_displacement_significant(obj_width, obj_height, prev_point, curr_point):
    # interpolation[{6000,70},{36000,120},{114000,250}]
    # min_displacement = obj_width * obj_height / 600 + 60
    
    """min_displacement = math.sqrt(obj_width * obj_height) / 8  # TODO: TO TUNE
    vector = [curr_point[i] - prev_point[i] for i in range(len(curr_point))]
    actual_displacement = math.hypot(*vector)
    return actual_displacement >= min_displacement"""
    min_value = obj_width * obj_height / 36  # TODO: TO TUNE
    products = [curr_point[i] - prev_point[i] for i in range(len(curr_point))]
    products = [products[i] * products[i] for i in range(len(products))]
    actual_value = sum(products)
    return actual_value >= min_value

class DirectMoveState:
    Green = 0
    Yellow = 1
    Red = 2


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    history_length : int
        Max number of points in detection history.
    batch_size : int
        Number of points to find average point and to put in history.
    class_ : any type
        Class id of object detector or person id/vector or anything depend on
        tracker mode so tracker could not confuse different types of objects.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    history : deque
        History of avg points where the object was detected on previous frames.
    batch : list
        Batch for history where points waited to be averaged.
    class_ : any type
        Class id of object detector or person id/vector or anything depend on
        tracker mode so tracker could not confuse different types of objects.
    batch_size : int
        Number of points to find average point and to put in history.
    is_checked : boolean
        Has this object been checked for any events since last point was
        added to history
    has_crossed_line : boolean
        Whether tracked object crossed a line defined in module settings of
        object detector (tracked_mode == 1)
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 history_length, batch_size, class_, confidence, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = deque([], history_length)
        self.batch = []
        self.class_ = class_
        self.confidence = confidence
        self.batch_size = batch_size
        self.is_checked = False

        self.has_crossed_line = False
        self.is_in_forbidden_region = False
        self.direct_move_state = DirectMoveState.Green

        self.state = TrackState.Tentative
        self.is_new = True
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age


    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        # self.history.appendleft(detection.get_center_bottom())
        self.batch.append(detection.get_center_center())
        if len(self.batch) >= self.batch_size:
            curr_batch_size = len(self.batch)
            avg_point = self.batch[0]
            for i in range(1, curr_batch_size):
                for j in range(len(avg_point)):
                    avg_point[j] += self.batch[i][j]
            for i in range(len(avg_point)):
                avg_point[i] = int(avg_point[i] / curr_batch_size)

            tlwh = detection.to_tlwh()
            if len(self.history) > 0 and is_displacement_significant(tlwh[2],\
                    tlwh[3], avg_point, self.history[0]) or\
                    len(self.history) == 0:
                self.history.appendleft(avg_point)
                self.is_checked = False
            self.batch.clear()
            self.batch.append(detection.get_center_center())

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def should_be_drawn(self):
        return self.is_confirmed() and self.time_since_update <= 1

    def get_last_position(self):
        if len(self.batch) > 0:
            return self.batch[-1]
        elif len(self.history) > 0:
            return self.history[0]
        else:
            return None
