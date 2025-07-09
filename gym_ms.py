import gymnasium as gym
from gymnasium import spaces
import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Mars, Venus, Jupiter, Saturn, Sun
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit

from numpy.linalg import norm
# from numpy import cross, cos, sin

class Gravity_cleanup(gym.Env):
    def __init__(self, env_config=None):

        if env_config is None:
            env_config = {}
        self.max_total_dv = env_config.get("max_total_dv", 250000.0)  # m/s removed total dv from json, will add again later, for now use backup value

        start_epoch = env_config.get("start_epoch", "2020-05-01") #collects start epoch from json file, otherwise uses backupdate given
        self.initial_epoch = Time(start_epoch, scale="tdb")
        self.epoch = self.initial_epoch

        self.max_dv = 100.0  # m/s per action
        self.step_duration = 1 * u.day 

        duration_days = env_config.get("max_duration", 804) #same here from json
        self.max_duration = duration_days * u.day

        self.planet_names = ["VENUS", "MARS", "JUPITER", "SATURN"]
        self.planets = {}
        for name in self.planet_names + ["EARTH"]:
            body = eval(name.title())
            epoch_offset = 0 # removed offset for final planet, might readd again to get final positon
            self.planets[name] = self._get_orbit(body, self.epoch + epoch_offset)

        self.target = self.planets["MARS"]
        self.start = self.planets["EARTH"]

        self.MAX_POS = 4e8 * u.km #scaling values to normalize observations
        self.MAX_VEL = 50000 * u.m / u.s  # m/s #scaling values to normalize observations

        self.flyby_radii = {}
        self._compute_flyby_radii()

        self.flyby_altitudes = { #not used currently
            "VENUS": 400 * u.km,
            "MARS": 500 * u.km,
            "JUPITER": 8000 * u.km,
            "SATURN": 12000 * u.km,
        }

        self.total_dv_used = 0.0
        self.episode_over = False

        self.observation_space = spaces.Box(low=-1e3, high=1e3, shape=(13,), dtype=np.float32) #
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) #normalized action space, scaled later to dv max 
 
        default_weights = { #backup weights
            "alignment": 1.0,
            "approach": 1.0,
            "flyby": 1.0,
            "dv_penalty": 1.0,
            "closing_rate": 1.0,
            "energy": 1.0,
            "distance": 1.0,
            "sun_penalty": 1.0,
            "final_reward": 1.0,
            "max_dv_exceeded": 1.0,
            "true_anomaly": 1.0,
        }

        self.reward_weights = env_config.get("reward_weights", default_weights)

    def _get_orbit(self, body, epoch):
        ephem = Ephem.from_body(body, epoch)
        return Orbit.from_ephem(Sun, ephem, epoch=epoch)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_offset = np.random.uniform(-10, 10) * u.day
        self.epoch = self.initial_epoch + start_offset

        self.elapsed_time = 0 * u.day
        self.total_dv_used = 0.0
        self.episode_over = False

        self.planets = {
            name: self._get_orbit(eval(name.title()), self.epoch)
            for name in self.planet_names + ["EARTH"]
        }

        earth = self.planets["EARTH"]
        self.state_r = earth.r
        self.state_v = earth.v.to(u.m / u.s)

        return self._get_obs(), {}

    def _compute_flyby_radii(self):
        for name in self.planet_names:
            body = eval(name.title())
            orbit = self._get_orbit(body, self.epoch)
            a = orbit.a.to(u.km)
            m = body.k.to(u.km**3 / u.s**2)
            M = Sun.k.to(u.km**3 / u.s**2)
            soi = a * (m / M) ** (2 / 5)
            self.flyby_radii[name] = soi

    def _get_obs(self):
        try:
            obs = []

            v_scale = self.MAX_VEL.to_value(u.m / u.s)
            r_scale = self.MAX_POS.to_value(u.km)

            v_sc = self.state_v.to_value(u.m / u.s) / v_scale
            obs.extend(v_sc.tolist())

            r_sc = self.state_r.to_value(u.km) / r_scale
            obs.extend(r_sc.tolist())

            rel_to_target = (self.target.r - self.state_r).to_value(u.km) / r_scale
            obs.extend(rel_to_target.tolist())

            dist_to_target = norm(rel_to_target)
            obs.append(dist_to_target)

            normalized_time=(self.elapsed_time / self.max_duration).to_value()
            obs.append(normalized_time)
            current_orbit = Orbit.from_vectors(Sun, self.state_r, self.state_v, epoch=self.epoch)
            mars_orbit = self.target

            nu_sc = current_orbit.nu.to_value(u.rad)
            nu_mars = mars_orbit.nu.to_value(u.rad)

            obs.append(nu_sc)
            obs.append(nu_mars)
            return np.array(obs, dtype=np.float32)

        except Exception as e:
            print("Exception during _get_obs (normalized):", e)
            return np.zeros(13, dtype=np.float32)

    def step(self, action):
        if self.episode_over:
            truncated = self.elapsed_time >= self.max_duration
            return self._get_obs(), 0.0, True, truncated, {}
        self.target = self._get_orbit(Mars, self.epoch)
        #print(action)
        delta_v= action*self.max_dv *u.m/u.s
        #print(delta_v)
        #delta_v =np.clip(action, -self.max_dv, self.max_dv) * u.m / u.s
        self.last_dv = delta_v.to_value(u.m / u.s)
        self.total_dv_used += norm(delta_v.to_value(u.m / u.s)) 

        new_velocity = self.state_v + delta_v
        orbit = Orbit.from_vectors(Sun, self.state_r, new_velocity, epoch=self.epoch)
        propagated_orbit = orbit.propagate(self.step_duration)

        self.state_r = propagated_orbit.r
        self.state_v = propagated_orbit.v.to(u.m / u.s)
        self.epoch = propagated_orbit.epoch
        self.elapsed_time += self.step_duration

        alignment_reward = 0.0
        approach_reward_total = 0.0
        flyby_reward = 0.0
        dv_penalty = -norm(delta_v.to_value(u.m / u.s))*1e-3
        closing_rate_reward = 0.0
        energy_reward = 0.0
        distance_penalty = 0.0
        sun_penalty = 0.0
        final_reward = 0.0
        max_dv_exceeded_penalty = 0.0
        true_anomaly_reward = 0.0

        max_s = self.max_duration.to(u.s).value
        elapsed_s = self.elapsed_time.to(u.s).value

        current_orbit = Orbit.from_vectors(Sun, self.state_r, self.state_v, epoch=self.epoch)
        #target_energy = abs(self.target.energy.to_value(u.km**2 / u.s**2))

        #energy_error = abs(current_orbit.energy - self.target.energy).to(u.km**2 / u.s**2).value
        #relative_error = energy_error / target_energy
        ##k=0.5
        #energy_reward = float(np.exp(-k * relative_error))
        #energy_reward =   (1.0 / (1.0 + relative_error))
        
        start_energy = self.start.energy.to(u.km**2 / u.s**2)
        target_energy = self.target.energy.to(u.km**2 / u.s**2)
        current_energy = current_orbit.energy.to(u.km**2 / u.s**2)

        initial_gap = abs(start_energy - target_energy).value
        current_gap = abs(current_energy - target_energy).value
      
        # 0 when at Earth energy, 1 when matching Mars energy
        progress = 1.0 - current_gap / initial_gap
        energy_reward = progress

        # if relative_error < 0.2:
        #     nu_sc = current_orbit.nu.to_value(u.rad)
        #     nu_target = self.target.nu.to_value(u.rad)
        #     angle_diff = np.arctan2(np.sin(nu_sc - nu_target), np.cos(nu_sc - nu_target))
        #     angle_error = abs(angle_diff)
        #     true_anomaly_reward = 1.0 / (1.0 + angle_error)

        distance_to_target = norm((self.state_r - self.target.r).to_value(u.km))
        distance_reward = (1 - (distance_to_target / self.MAX_POS.to_value(u.km))) * (elapsed_s / max_s)

        dist_from_sun = norm(self.state_r.to_value(u.km))
        au_km = (1 * u.AU).to_value(u.km)
        if dist_from_sun < (0.5 * au_km):
            sun_penalty -= 1000

        done = distance_to_target < 1e5 or self.elapsed_time >= self.max_duration
        if done:
            final_reward += 1000000 / (1 + distance_to_target / 1e6)

        if self.total_dv_used > self.max_total_dv:
            done = True
            max_dv_exceeded_penalty = -5

        reward = (
            self.reward_weights["alignment"] * alignment_reward +
            self.reward_weights["approach"] * approach_reward_total +
            self.reward_weights["flyby"] * flyby_reward +
            self.reward_weights["dv_penalty"] * dv_penalty +
            self.reward_weights["closing_rate"] * closing_rate_reward +
            self.reward_weights["energy"] * energy_reward +
            self.reward_weights["distance"] * distance_reward +
            self.reward_weights["sun_penalty"] * sun_penalty +
            self.reward_weights["final_reward"] * final_reward +
            self.reward_weights["max_dv_exceeded"] * max_dv_exceeded_penalty +
            self.reward_weights["true_anomaly"] * true_anomaly_reward
        )

        self.episode_over = done
        # if done:
        #     print(f"TRAINING END: total Δv = {self.total_dv_used / 1000:.3f} km/s")


        info = dict(
            alignment=alignment_reward,
            approach=approach_reward_total,
            flyby=flyby_reward,
            dv_penalty=dv_penalty,
            closing_rate=closing_rate_reward,
            energy=energy_reward,
            distance=distance_reward,
            sun_penalty=sun_penalty,
            final_reward=final_reward,
            max_dv_exceeded=max_dv_exceeded_penalty,
            total_dv_used_kms=self.total_dv_used / 1000.0,  # for backward compatibility
            final_distance_to_saturn_km=distance_to_target,
            true_anomaly=true_anomaly_reward
        )

        truncated = self.elapsed_time >= self.max_duration
        reward = float(reward)
        #reward=np.array([reward])
        return self._get_obs(), reward, done, truncated, info
