from abc import ABC, abstractmethod
import json
import xarray as xr
import numpy as np
import pandas as pd
import fast_optimization as fo
from scipy.stats import circmean
from .waves import BreakingPropagation


class CoastlineModel(ABC):
    """
    Abstract base class for coastline models.
    """

    def __init__(self, 
                 path: str,
                 model_name: str, 
                 mode: str, 
                 model_type: str, 
                 model_key: str):
        """
        Initialize the CoastlineModel with a path to the model data.
        
        :param path: Path to the model data file.
        """
        self.path = path
        self._load_data(model_key)
        
        self.name = model_name
        self.mode = mode
        self.type = model_type
        
        self._set_type()
        self._setup_mode()
        self._compute_time_step()
        # verify if cfg·['switch_brk'] exists
        if 'switch_brk' in self.cfg:
            self._break_waves_snell()

    def _set_type(self):
        """
        Set up the type of coastline model based on the configuration.
        """
        if self.type == 'CS':
            self._setup_crossshore_vars()
        elif self.type == 'RT':
            self._setup_rotation_vars()
            
        # else self.type == 'OL':
        #     self._setup_oneline_vars()
        # else:
        #     raise ValueError(f"Unknown type: {self.type}")
    def _setup_mode(self):
        """
        Set up the model based on the specified mode.
        """
        if self.mode == 'calibration':
            self.cal_alg = self.cfg['cal_alg']
            self.metrics = self.cfg['metrics']
            self.lb = self.cfg['lb']
            self.ub = self.cfg['ub']
            self.calibr_cfg = fo.config_cal(self.cfg)
            self._split_data_c()
        elif self.mode == 'standalone':
            self._split_data_dr()
            
    def _split_data_c(self):
        """
        Split the dataset into training and validation sets based on the time range.
        """
        ii = np.where(self.time>=self.start_date)[0][0]
        self.time = self.time[ii:]
        jj = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.time_s = self.time[jj]
        kk = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]

        self.idx_validation     = np.where((self.time < self.start_date) | (self.time > self.end_date))[0]
        self.idx_calibration    = jj
        self.idx_validation_obs = np.where((self.time_obs < self.start_date) | (self.time_obs > self.end_date))[0]
        
        self._split_cal_vars(ii, jj, kk)

    def _split_data_dr(self):
        """
        Split the dataset into training and validation sets based on the time range.
        """
        ii = np.where((self.time >= self.start_date) & (self.time <= self.end_date))[0]
        self.time = self.time[ii]
        self.hs   = self.hs[ii]
        self.tp   = self.tp[ii]
        self.dir  = self.dir[ii]
        self.tide = self.tide[ii]
        self.surge = self.surge[ii]
        jj = np.where((self.time_obs >= self.start_date) & (self.time_obs <= self.end_date))[0]
        self.time_s = self.time[jj]
        self.Obs = self.Obs[jj]
        self.time_obs = self.time_obs[jj]

    def _split_cal_vars(self, ii, jj, kk):
        """
        Set up variables for calibration and validation based on the indices.
        
        :param ii: Index for the start date.
        :param jj: Indices for the calibration period.
        :param kk: Indices for the observation period.
        :param mm: Indices for the validation observations.
        """
        
        self.hs   = self.hs[ii:]
        self.tp   = self.tp[ii:]
        self.dir  = self.dir[ii:]
        self.tide = self.tide[ii:]
        self.surge = self.surge[ii:]

        self.hs_s   = self.hs[jj]
        self.tp_s   = self.tp[jj]
        self.dir_s  = self.dir[jj]
        self.tide_s = self.tide[jj]
        self.surge_s = self.surge[jj]

        self.Obs_splited  = self.Obs[kk]
        self.time_obs_s   = self.time_obs[kk]

        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time_s - t)))
        self.idx_obs_splited = mkIdx(self.time_obs_s)

        # Validation
        if len(self.idx_validation)>0:
            mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time[self.idx_validation] - t)))
            if len(self.idx_validation_obs)>0:
                self.idx_validation_for_obs = mkIdx(self.time_obs[self.idx_validation_obs])
            else:
                self.idx_validation_for_obs = []
        else:
            self.idx_validation_for_obs = []
        

    def _load_data(self, jkey: str):
        """ 
        Load the dataset from the specified path.
        """
        self.data = xr.open_dataset(self.path)
        self.data.load()
        self.cfg = json.loads(self.data.attrs[jkey])
        self.start_date = pd.to_datetime(self.cfg['start_date'])
        self.end_date = pd.to_datetime(self.cfg['end_date'])
        self.time = pd.to_datetime(self.data.time.values)
        self.time_obs = pd.to_datetime(self.data.time_obs.values)
        self.hs = self.data.hs.values
        self.tp = self.data.tp.values
        self.dir = self.data.dir.values
        self.tide = self.data.tide.values
        self.surge = self.data.surge.values
        self.Obs = self.data.obs.values
        self.Obs_avg = self.data.average_obs.values
        self.rot = self.data.rot.values
        self.mask_nan_rot = self.data.mask_nan_rot.values
        self.mask_nan_obs = self.data.mask_nan_obs.values
        self.mask_nan_average_obs = self.data.mask_nan_average_obs.values
        self.depth = self.data.waves_depth.values
        self.bathy_angle = self.data.phi.values
        self.x_pivotal = self.data.x_pivotal.values
        self.y_pivotal = self.data.y_pivotal.values
        self.xi = self.data.xi.values
        self.yi = self.data.yi.values
        self.xf = self.data.xf.values
        self.yf = self.data.yf.values
        self.data.close()

    def _setup_crossshore_vars(self):
        """
        Set up cross-shore variables for the model.
        """
        if self.cfg['trs'] == 'Average':
            self.hs = np.mean(self.hs, axis=1)
            self.tp = np.mean(self.tp, axis=1)
            self.dir = circmean(self.dir, axis=1)
            self.tide = np.mean(self.tide, axis=1)
            self.surge = np.mean(self.surge, axis=1)
            self.Obs = self.Obs_avg
            self.Obs = self.Obs[~self.mask_nan_average_obs]
            self.time_obs = self.time_obs[~self.mask_nan_average_obs]
            self.depth = np.mean(self.depth)
            self.bathy_angle = circmean(self.bathy_angle)
        elif isinstance(self.cfg['trs'], int):
            self.hs = self.hs[:, self.cfg['trs']]
            self.tp = self.tp[:, self.cfg['trs']]
            self.dir = self.dir[:, self.cfg['trs']]
            self.tide = self.tide[:, self.cfg['trs']]
            self.surge = self.surge[:, self.cfg['trs']]
            self.Obs = self.Obs[:, self.cfg['trs']]
            self.Obs = self.Obs[~self.mask_nan_obs[:, self.cfg['trs']]]
            self.time_obs = self.time_obs[~self.mask_nan_obs[:, self.cfg['trs']]]
            self.depth = self.depth[self.cfg['trs']]
            self.bathy_angle = self.bathy_angle[self.cfg['trs']]



    def _setup_rotation_vars(self):
        """
        Set up rotation variables for the model.
        """
        trs, dists_ = self._find_two_closest_transects()
        self._interpolate_rot_vars(trs, dists_)


    def _compute_time_step(self):
        """
        Compute the time step for the given time array.
        
        :param time: Array of time values.
        :return: Time step in seconds.
        """
        
        mkIdx = np.vectorize(lambda t: np.argmin(np.abs(self.time - t)))

        self.idx_obs = mkIdx(self.time_obs)
        # Now we calculate the dt from the time variable
        mkDT = np.vectorize(lambda i: (self.time[i+1] - self.time[i]).total_seconds()/3600)
        self.dt = mkDT(np.arange(0, len(self.time)-1))
        if self.mode == 'calibration':
            mkDTsplited = np.vectorize(lambda i: (self.time_s[i+1] - self.time_s[i]).total_seconds()/3600)
            self.dt_s = mkDTsplited(np.arange(0, len(self.time_s)-1))

    def _break_waves_snell(self):
        """
        Break waves using Snell's law.
        """
        if self.cfg['switch_brk'] == 1:
            self.break_type = self.cfg['break_type']
            self.hb, self.dirb, self.depthb = BreakingPropagation(self.hs, self.tp, self.dir, np.repeat(self.depth, len(self.hs)), np.repeat(self.bathy_angle, len(self.hs)), self.break_type)
            if self.mode == 'calibration':
                self.hb_s, self.dirb_s, self.depthb_s = BreakingPropagation(self.hs_s, self.tp_s, self.dir_s, np.repeat(self.depth, len(self.hs_s)), np.repeat(self.bathy_angle, len(self.hs_s)), self.break_type)
        else:
            self.hb = self.hs
            self.dirb = self.dir
            self.depthb = self.hb/0.55
            if self.mode == 'calibration':
                self.hb_s = self.hs_s
                self.dirb_s = self.dir_s
                self.depthb_s = self.hb_s/0.55

    @abstractmethod
    def setup_forcing(self):
        """Prepare forcing arrays (e.g., E, P, E_s, P_s, Yini)."""
        pass

    @abstractmethod
    def run_model(self, par: np.ndarray) -> np.ndarray:
        """Run full model over all timesteps."""
        pass

    def calibrate(self):
        """Generic calibration flow using fast_optimization."""
        sol, objs, hist = self.calibr_cfg.calibrate(self)
        if getattr(self, 'is_exp', False):
            sol = np.exp(sol)
        self.solution, self.objectives, self.hist = sol, objs, hist
        self.run(self.solution)
        
    def run(self, par: np.ndarray) -> np.ndarray:
        """
        Run the model with the given parameters.
        
        :param par: Parameters for the model.
        :return: Model output.
        """
        self.par_values = par
        self._set_parameter_names()
        self.full_run = self.run_model(self.par_values)
        

    @abstractmethod
    def _set_parameter_names(self):
        """Assign self.par_names and self.par_values after calibration."""
        pass

    def _interpolate_rot_vars(self, trs, dists_):
        """
        Interpolate rotation variables based on the closest transects and distances.
        
        :param trs: Indices of the closest transects.
        :param dists_: Distances to the transects.
        """
        self.hs = interpolate_by_distance(self.hs[:, trs], dists_)
        self.tp = interpolate_by_distance(self.tp[:, trs], dists_)
        self.dir = interpolate_by_distance(self.dir[:, trs], dists_)
        self.tide = interpolate_by_distance(self.tide[:, trs], dists_)
        self.surge = interpolate_by_distance(self.surge[:, trs], dists_)
        self.Obs = self.rot[~self.mask_nan_rot]
        self.time_obs = self.time_obs[~self.mask_nan_rot]
        self.depth = interpolate_by_distance(self.depth[trs], dists_)[0]
        self.bathy_angle = interpolate_by_distance(self.bathy_angle[trs], dists_)[0]

    def _find_two_closest_transects(self):
        
        """
        Find the indices and distances of the two transects closest to a pivot point.

        Parameters:
        - X0, Y0, Xf, Yf: sequences of equal length containing start and end coordinates for each transect.
        - pivot: tuple (px, py) representing the reference point.

        Returns:
        - indices: list of the two indices with the smallest distances.
        - distances: list of the corresponding distances to the pivot point.
        """
        px, py = self.x_pivotal, self.y_pivotal
        X0, Y0 = self.xi, self.yi
        Xf, Yf = self.xf, self.yf
        n = len(X0)
        distances = []
        for i in range(n):
            d = self._point_to_segment_distance(px, py, X0[i], Y0[i], Xf[i], Yf[i])
            distances.append(d)
        distances = np.array(distances).squeeze()

        # Sort distances and return the two smallest
        idx_sorted = np.argsort(distances)
        return idx_sorted[:2].tolist(), distances[idx_sorted[:2]].tolist()


    def _point_to_segment_distance(self, px, py, x0, y0, xf, yf):
        """
        Calculate the minimum distance between a point and a line segment.

        Parameters:
        - px, py: coordinates of the point.
        - x0, y0: coordinates of the segment start.
        - xf, yf: coordinates of the segment end.

        Returns:
        - The shortest Euclidean distance from point (px, py) to the segment.
        """
        # Direction vector of the segment and vector from start to point
        sx, sy = xf - x0, yf - y0
        vx, vy = px - x0, py - y0

        # Square length of the segment
        seg_len2 = sx**2 + sy**2
        if seg_len2 == 0:
            # Segment is a single point
            return np.hypot(vx, vy)

        # Project v onto s normalized by |s|^2
        t = (vx * sx + vy * sy) / seg_len2
        t = np.clip(t, 0, 1)

        # Coordinates of the projection onto the segment
        proj_x = x0 + t * sx
        proj_y = y0 + t * sy

        # Euclidean distance between point and its projection
        return np.hypot(px - proj_x, py - proj_y)

def interpolate_by_distance(H, distances):
    """
    Interpolate between two sets of values based on their distances to a pivot point.

    Parameters:
    - H: array-like of shape (m, 2), where column 0 is values from the first transect
         and column 1 is values from the second transect.
    - distances: sequence of two non-negative distances [d1, d2].

    Returns:
    - numpy.ndarray of shape (m,), interpolated values elementwise, weighted inversely by distance.
      If d1 or d2 is zero, returns the corresponding column from H.
    """
    if len(H.shape) == 1:
        H = H.reshape(-1, 2)
    H = np.array(H, dtype=float)
    d = np.array(distances, dtype=float)
    if d[0] == 0:
        return H[:, 0].copy()
    if d[1] == 0:
        return H[:, 1].copy()
    w = 1 / d
    w /= w.sum()
    return w[0] * H[:, 0] + w[1] * H[:, 1]


