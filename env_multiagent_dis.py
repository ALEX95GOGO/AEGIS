"""
Car-following environment wrapper for CARLA with pygame visualization.

"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard Library
# -----------------------------------------------------------------------------
import argparse
import collections
import csv
import glob
import math
import os
import random
import re
import socket
import sys
import time
import weakref
from datetime import datetime

# -----------------------------------------------------------------------------
# Third-Party
# -----------------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt  # noqa: F401 (kept for parity with original)
import numpy as np
import pygame
import pygame.freetype
from pygame import gfxdraw  # noqa: F401 (kept for parity with original)
from pygame.locals import (
    KMOD_CTRL,
    KMOD_SHIFT,
    K_TAB,
    K_c,
    K_d,
    K_a,
    K_DOWN,
    K_ESCAPE,
    K_LEFT,
    K_RIGHT,
    K_UP,
    K_q,
    K_s,
    K_w,
)

# -----------------------------------------------------------------------------
# CARLA Imports (path hack preserved)
# -----------------------------------------------------------------------------
try:
    sys.path.append(
        glob.glob(
            r"E:\UTS\code\CARLA_9_13\carla\PythonAPI\carla\dist\carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla  # type: ignore
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: F401 (kept)
from agents.navigation.local_planner import LocalPlanner, RoadOption  # noqa: F401
from carla import ColorConverter as cc
from carla import Location
from carla import Transform, Rotation

# -----------------------------------------------------------------------------
# Config parser import (Py2/3)
# -----------------------------------------------------------------------------
if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:  # pragma: no cover
    from ConfigParser import RawConfigParser as ConfigParser  # type: ignore

# -----------------------------------------------------------------------------
# Display / Render constants
# -----------------------------------------------------------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 320
WIDTH, HEIGHT = 80, 45

FULLGREEN = (0, 255, 0)
FULLRED = (255, 0, 0)
FULLBLUE = (0, 0, 255)
FULLBLACK = (255, 255, 255)

SENSORS_LIMITS = {
    "sensor.camera.rgb": 10,
    "sensor.camera.semantic_segmentation": 10,
    "sensor.camera.depth": 10,
    "sensor.lidar.ray_cast": 1,
    "sensor.lidar.ray_cast_semantic": 1,
    "sensor.other.radar": 2,
    "sensor.other.gnss": 1,
    "sensor.other.imu": 1,
    "sensor.opendrive_map": 1,
    "sensor.speedometer": 1,
}


class CarFollowing(object):
    """
    Car-following scenario environment built on CARLA.

    NOTE:
    - Public API (method names and signatures) preserved.
    - No functional changes; formatting/comments/docstrings only.
    """

    def __init__(
        self,
        enabled_obs_number: int = 2,
        vehicle_type: str = "random",
        joystick_enabled: bool = False,
        control_interval: int = 1,
        conservative_surrounding: bool = False,
        frame: int = 12,
        port: int = 2000,
        mode: str = "auto",
        town: int = 7,
    ):
        # --- UDP for event markers ---
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 9999
        random.seed(42)
        print(f"UDP target IP: {self.UDP_IP}")
        print(f"UDP target port: {self.UDP_PORT}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # --- Observation / Action specs ---
        self.observation_size_width = WIDTH
        self.observation_size_height = HEIGHT
        self.observation_size = WIDTH * HEIGHT
        self.action_size = 1
        self.mode = mode

        # --- CARLA world parameters / switches ---
        self.vehicle_type = vehicle_type
        self.joystick_enabled = joystick_enabled
        self.intervention_type = "joystick" if joystick_enabled else "keyboard"
        self.control_interval = control_interval
        self.conservative_surrounding = conservative_surrounding

        # --- Vehicles / Actors ---
        self.ego_vehicle = None
        self.front_vehicle = None
        self.back_vehicle = None
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [], []
        self.enabled_obs = enabled_obs_number

        # --- Sensors ---
        self.collision_sensor = None
        self.seman_camera = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros((int(720 / 2), int(1280 / 2), 3))
        self.recording = False
        self.Attachment = carla.AttachmentType

        # --- CARLA client / world ---
        self.port = port
        self.client = carla.Client("localhost", port)
        self.client.set_timeout(60.0)
        self.brake_idx = 300  # initial time to brake
        self.town = town
        self.frame = frame
        self.travel_dis = 0

        # --- Load world by town id ---
        if self.town == 4:
            self.world = self.client.load_world("Town04")
        if self.town == 40:
            self.world = self.client.load_world("Town04")
        if self.town == 5:
            self.world = self.client.load_world("Town05")
        if self.town == 7:
            self.world = self.client.load_world("Town07")
        if self.town == 6:
            self.world = self.client.load_world("Town06")
        if self.town == 3:
            self.world = self.client.load_world("Town03")
        if self.town == 2:
            self.world = self.client.load_world("Town02")
        if self.town == 10:
            self.world = self.client.load_world("Town10HD")

        self.map = self.world.get_map()

        # --- Weather ---
        self._weather_presets = find_weather_presets()
        self._weather_index = 2
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])

        # --- Timestamp suffix for logs ---
        self.now = datetime.now().strftime("-%d-%m-%Y-%H-%M-%S")

        # --- Pygame UI init ---
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.infoObject = pygame.display.Info()
        pygame.display.set_caption("CarFollowing Scenario")

        # --- UI text labels ---
        font = pygame.font.Font("freesansbold.ttf", 32)
        self.text_humanguidance = font.render(
            "Human Guidance Mode", True, FULLBLACK, FULLGREEN
        )
        self.text_humanguidance_rect = self.text_humanguidance.get_rect()
        self.text_humanguidance_rect.center = (1000, 60)

        self.text_RLinference = font.render(
            "RL Inference Mode", True, FULLBLACK, FULLRED
        )
        self.text_RLinference_rect = self.text_humanguidance.get_rect()
        self.text_RLinference_rect.center = (1000, 60)

        self.text_humanmodelguidance = font.render(
            "Human Model Guidance Mode", True, FULLBLACK, FULLBLUE
        )
        self.text_humanmodelguidance_rect = self.text_humanguidance.get_rect()
        self.text_humanmodelguidance_rect.center = (1000, 60)

        self.random_list = random.sample(range(10001), 60)

        # --- Joystick setup (optional) ---
        if self.joystick_enabled:
            pygame.joystick.init()
            self._parser = ConfigParser()
            self._parser.read("./wheel_config.ini")
            self._steer_idx = int(self._parser.get("G29 Racing Wheel", "steering_wheel"))
            self._throttle_idx = int(self._parser.get("G29 Racing Wheel", "throttle"))
            self._brake_idx = int(self._parser.get("G29 Racing Wheel", "brake"))
            self._reverse_idx = int(self._parser.get("G29 Racing Wheel", "reverse"))
            self._handbrake_idx = int(self._parser.get("G29 Racing Wheel", "handbrake"))

        # self.reset()  # left disabled, call explicitly

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset world, spawn vehicles and sensors, and return initial observation."""
        self.collision = 0
        self.brake_idx = random.randint(200, 400)  # initial time to brake
        self.travel_dis = 0

        # World sync settings
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame
        self.world.apply_settings(settings)

        self.rgb = None

        # Recording lists
        self.intervene_history = []
        self.previous_action_list = []

        # Human intervention state
        self.intervention = False
        self.keyboard_intervention = False
        self.joystick_intervention = False
        self.risk = None
        self.v_upp = 19.5 / 7
        self.v_low = 13.5 / 7
        self.terminate_position = 0
        self.index_obs_concerned = None
        self.brake_flag = 0

        # Spawn ego vehicle (fixed)
        bp_ego = self.world.get_blueprint_library().filter("vehicle.audi.tt")[0]
        bp_back = self.world.get_blueprint_library().filter("vehicle.audi.etron")[0]
        bp_ego.set_attribute("color", "0, 0, 0")

        waypoints = self.world.get_map().get_spawn_points()
        waypoint_tuple_list = self.map.get_topology()  # noqa: F841

        if self.ego_vehicle is not None:
            self.destroy()

        spawn_point_ego = self.world.get_map().get_spawn_points()[0]
        self.ego_vehicle = self.world.spawn_actor(bp_ego, spawn_point_ego)

        # Town-specific ego transforms
        if self.town == 4:
            new_location = Location(x=215, y=-371, z=0.1)
            self.ego_vehicle.set_location(new_location)
            new_yaw_degrees = 90.0
            new_yaw_radians = new_yaw_degrees * (3.14159265359 / 180.0)
            current_transform = self.ego_vehicle.get_transform()
            new_rotation = Rotation(
                pitch=current_transform.rotation.pitch,
                yaw=new_yaw_radians,
                roll=current_transform.rotation.roll,
            )
            new_transform = Transform(location=new_location, rotation=new_rotation)
            self.ego_vehicle.set_transform(new_transform)

        if self.town == 40:
            new_location = Location(x=215, y=-368, z=0.1)
            self.ego_vehicle.set_location(new_location)
            new_yaw_degrees = 90.0
            new_yaw_radians = new_yaw_degrees * (3.14159265359 / 180.0)
            current_transform = self.ego_vehicle.get_transform()
            new_rotation = Rotation(
                pitch=current_transform.rotation.pitch,
                yaw=new_yaw_radians,
                roll=current_transform.rotation.roll,
            )
            new_transform = Transform(location=new_location, rotation=new_rotation)
            self.ego_vehicle.set_transform(new_transform)

        ego_location = self.ego_vehicle.get_location()
        print(
            f"Ego Vehicle Location: (x={ego_location.x}, y={ego_location.y}, z={ego_location.z})"
        )

        # Town-specific front/back spawns
        if self.town == 4:
            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.x = 225
            spawn_point_front.location.y = -371
            spawn_point_front.location.z = 0.1
            spawn_point_front.rotation.yaw = 0

            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.x = 185
            spawn_point_back.location.y = -371
            spawn_point_back.location.z = 0.1
            spawn_point_back.rotation.yaw = 0

        if self.town == 40:
            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.x = 225
            spawn_point_front.location.y = -368
            spawn_point_front.location.z = 0.1
            spawn_point_front.rotation.yaw = 0

            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.x = 185
            spawn_point_back.location.y = -368
            spawn_point_back.location.z = 0.1
            spawn_point_back.rotation.yaw = 0

            spawn_point_2 = self.world.get_map().get_spawn_points()[0]
            spawn_point_2.location.x = 235
            spawn_point_2.location.y = -374
            spawn_point_2.location.z = 0.1
            spawn_point_2.rotation.yaw = 0
            self.vehicle_2 = self.world.spawn_actor(bp_back, spawn_point_2)
            initial_velocity = carla.Vector3D(5, 0, 0)
            self.vehicle_2.set_target_velocity(initial_velocity)
            self.agent_2 = BasicAgent(self.vehicle_2, target_speed=5 * 3.6)

        if self.town == 5:
            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.x = -184
            spawn_point_front.location.y = -95
            spawn_point_front.location.z = 0.1
            spawn_point_front.rotation.yaw = 180

            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.x = -144
            spawn_point_back.location.y = -95
            spawn_point_back.location.z = 0.1
            spawn_point_back.rotation.yaw = 180

        if self.town == 7:
            new_x = self.world.get_map().get_spawn_points()[72].location.x + 10
            new_y = self.world.get_map().get_spawn_points()[72].location.y - 10
            new_z = 0.1

            spawn_point_2 = self.world.get_map().get_spawn_points()[0]
            spawn_point_2.location.x = new_x - 2
            spawn_point_2.location.y = new_y - 25
            spawn_point_2.location.z = 0.1
            spawn_point_2.rotation.yaw = 45

            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.x = new_x - 10
            spawn_point_back.location.y = new_y + 10
            spawn_point_back.location.z = 0.1
            spawn_point_back.rotation.yaw = -90

            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.x = new_x + 2
            spawn_point_front.location.y = new_y - 20
            spawn_point_front.location.z = 0.1
            spawn_point_back.rotation.yaw = 45

            new_location = Location(x=new_x, y=new_y, z=new_z)
            self.ego_vehicle.set_location(new_location)
            new_yaw_degrees = -90  # CARLA uses degrees
            current_transform = self.ego_vehicle.get_transform()
            new_rotation = Rotation(
                pitch=current_transform.rotation.pitch,
                yaw=new_yaw_degrees,
                roll=current_transform.rotation.roll,
            )
            new_transform = Transform(location=new_location, rotation=new_rotation)
            self.ego_vehicle.set_transform(new_transform)

        if self.town == 3:
            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.x = -6.4
            spawn_point_front.location.y = -49
            spawn_point_front.location.z = 0.1
            spawn_point_front.rotation.yaw = 90

            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.x = -6.4
            spawn_point_back.location.y = -99
            spawn_point_back.location.z = 0.1
            spawn_point_back.rotation.yaw = 90

        if self.town == 2:
            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.y = 172
            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.y = 122

        if self.town == 6:
            spawn_point_front = self.world.get_map().get_spawn_points()[0]
            spawn_point_front.location.x = (
                self.world.get_map().get_spawn_points()[0].location.x - 20
            )
            spawn_point_front.location.y = (
                self.world.get_map().get_spawn_points()[0].location.y - 7
            )
            spawn_point_back = self.world.get_map().get_spawn_points()[0]
            spawn_point_back.location.x = (
                self.world.get_map().get_spawn_points()[0].location.x + 20
            )
            spawn_point_back.location.y = (
                self.world.get_map().get_spawn_points()[0].location.y - 7
            )
            new_location = Location(
                x=ego_location.x, y=ego_location.y - 7, z=ego_location.z
            )
            self.ego_vehicle.set_location(new_location)

        # Spawn front / back
        self.front_vehicle = self.world.spawn_actor(bp_ego, spawn_point_front)
        self.back_vehicle = self.world.spawn_actor(bp_back, spawn_point_back)
        self.ego_location = self.ego_vehicle.get_location()
        self.y_ego_old = None
        self.x_ego_old = None
        self.world.tick()

        # Traffic manager (sync)
        self.tm = self.client.get_trafficmanager(6000 + self.port)
        self.tm.set_synchronous_mode(True)
        tm_port = self.tm.get_port()  # noqa: F841 (kept)

        # Agents
        self.agent = BasicAgent(self.ego_vehicle, target_speed=8 * 3.6)
        self.front_agent = BasicAgent(
            self.front_vehicle,
            target_speed=18 * 3.6,
            opt_dict={
                "ignore_stop_signs": True,
                "ignore_vehicles": True,
                "ignore_traffic_lights": True,
            },
        )
        self.back_agent = BasicAgent(
            self.back_vehicle,
            target_speed=18 * 3.6,
            opt_dict={
                "ignore_stop_signs": True,
                "ignore_vehicles": True,
                "ignore_traffic_lights": True,
            },
        )

        initial_velocity = carla.Vector3D(5, 0, 0)
        self.front_vehicle.set_target_velocity(initial_velocity)
        self.ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

        # Vehicle control
        self.control = carla.VehicleControl()
        self.h_control = self.control  # human controller
        self.heuristic = 330.5

        # Collision sensor
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
            bp_collision, carla.Transform(), attach_to=self.ego_vehicle
        )
        weak_self = weakref.ref(self)
        # self.collision_sensor.listen(lambda event: CarFollowing._on_collision(weak_self, event))

        # Cameras
        self.camera_transforms = [
            (
                carla.Transform(carla.Location(x=-1, z=50), carla.Rotation(pitch=0)),
                self.Attachment.SpringArm,
            ),
            (
                carla.Transform(carla.Location(x=-5, z=50), carla.Rotation(pitch=0)),
                self.Attachment.SpringArm,
            ),
        ]
        self.camera_transform_index = 1
        self.cameras = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB", {}],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
                {},
            ],
        ]

        # RGB camera blueprint
        bp_viz_camera = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp_viz_camera.set_attribute("image_size_x", "640")
        bp_viz_camera.set_attribute("image_size_y", "360")
        bp_viz_camera.set_attribute("sensor_tick", "0.04")
        bp_viz_camera.set_attribute("fov", "90")
        self.cameras[0].append(bp_viz_camera)

        # Semantic camera blueprint
        bp_seman_camera = self.world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        bp_seman_camera.set_attribute("image_size_x", "80")
        bp_seman_camera.set_attribute("image_size_y", "45")
        bp_seman_camera.set_attribute("sensor_tick", "0.04")
        bp_viz_camera.set_attribute("fov", "90")
        self.cameras[1].append(bp_seman_camera)

        # Spawn cameras
        if self.seman_camera is not None:
            self.seman_camera.destroy()
            self.viz_camera.destroy()
            self.surface = None

        self.viz_camera = self.world.spawn_actor(
            self.cameras[0][-1],
            carla.Transform(carla.Location(x=2.8, z=1.7)),
            attach_to=self.ego_vehicle,
        )
        self.seman_camera = self.world.spawn_actor(
            self.cameras[1][-1],
            carla.Transform(carla.Location(x=2.8, z=1.7)),
            attach_to=self.ego_vehicle,
        )

        # Camera listeners
        weak_self = weakref.ref(self)
        self.seman_camera.listen(
            lambda image: CarFollowing._parse_seman_image(weak_self, image)
        )
        self.viz_camera.listen(
            lambda image: CarFollowing._parse_image(weak_self, image)
        )

        # Step counter
        self.count = 0

        # Initial observation
        state, _ = self.get_observation()

        # Distance helpers
        _ = (
            (self.back_vehicle.get_location().x - self.front_vehicle.get_location().x)
            ** 2
            + (
                self.back_vehicle.get_location().y
                - self.front_vehicle.get_location().y
            )
            ** 2
        ) ** 0.5

        _ = (
            (self.ego_vehicle.get_location().x - self.front_vehicle.get_location().x)
            ** 2
            + (self.ego_vehicle.get_location().y - self.front_vehicle.get_location().y)
            ** 2
        ) ** 0.5

        spectator = self.world.get_spectator()  # noqa: F841
        self.spawn_points = self.world.get_map().get_spawn_points()
        for j, spawn_point in enumerate(self.spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(j), life_time=300)

        return state

    def render(self, display):
        """Blit the current camera surface to the pygame window."""
        if self.surface is not None:
            m = pygame.transform.smoothscale(
                self.surface,
                [int(self.infoObject.current_w), int(self.infoObject.current_h)],
            )
            display.blit(m, (0, 0))

    # -------------------------------------------------------------------------
    # Sensor callbacks
    # -------------------------------------------------------------------------

    def _parse_seman_image(weak_self, image):
        """Semantic camera callback."""
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[1][1])
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_output = array

    def _parse_image(weak_self, image):
        """RGB camera callback + HUD overlays."""
        self = weak_self()
        if not self:
            return
        image.convert(self.cameras[0][1])
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.rgb = array
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.intervention:
            self.surface.blit(self.text_humanguidance, self.text_humanguidance_rect)
        else:
            self.surface.blit(self.text_RLinference, self.text_RLinference_rect)

    def show_human_model_mode(self):
        """Overlay a 'human model guidance' banner."""
        self.surface.blit(self.text_humanmodelguidance, self.text_humanmodelguidance_rect)

    def _on_collision(weak_self, event):
        """Collision callback (kept but not hooked by default)."""
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)

    # -------------------------------------------------------------------------
    # Inspectors
    # -------------------------------------------------------------------------

    def get_collision_history(self):
        """Return aggregated collision intensity per frame and a flag if any occurred."""
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(self, action):
        """
        Apply an action, advance the world one tick, and return the transition.
        Mirrors original logic; only formatting/comment polish applied.
        """
        self.world.tick()
        self.render(self.display)
        pygame.display.flip()

        human_control = None

        # Act once per control interval
        action = action[0] if action.shape != () else action
        action = (
            action
            if self.count % self.control_interval == 0
            else self.previous_action_list[-1]
        )

        simulation_time = self.world.get_snapshot().timestamp.elapsed_seconds  # noqa: F841

        # Agent policies
        self.control = self.agent.run_step()
        self.front_control = self.front_agent.run_step()
        self.back_control = self.back_agent.run_step()

        # Parse human inputs (pedal)
        if self.intervention_type == "joystick":
            human_control_pedal = self._parse_wheel()
        else:
            human_control_pedal = self._parse_key()

        self.intervention = self.joystick_intervention or self.keyboard_intervention

        # Parse human inputs (steer)
        if self.intervention_type == "joystick":
            human_control_steer = self._parse_wheel_steer()
        else:
            human_control_steer = self._parse_key_steer()

        # Control modes
        if self.mode == "manual":
            if self.intervention:
                self.control.throttle = float(abs(human_control_pedal)) * 1 if human_control_pedal >= 0 else 0
                self.control.brake = float(abs(human_control_pedal)) * 1 if human_control_pedal < 0 else 0
            else:
                self.control.throttle = 0
                self.control.brake = 0
            self.control.steer = human_control_steer if human_control_steer is not None else 0.001 * self.control.steer

        if self.mode == "steer":
            self.control.throttle = float(abs(action)) * 1 if action >= 0 else 0
            self.control.brake = float(abs(action)) * 1 if action < 0 else 0
            self.control.steer = human_control_steer if human_control_steer is not None else 0.001 * self.control.steer

        if self.mode == "pedal":
            if self.intervention:
                self.control.throttle = float(abs(human_control_pedal)) * 1 if human_control_pedal >= 0 else 0
                self.control.brake = float(abs(human_control_pedal)) * 1 if human_control_pedal < 0 else 0
            else:
                self.control.throttle = 0
                self.control.brake = 0

        if self.mode == "auto":
            if self.intervention:
                self.control.throttle = float(abs(human_control_pedal)) * 1 if human_control_pedal >= 0 else 0
                self.control.brake = float(abs(human_control_pedal)) * 1 if human_control_pedal < 0 else 0
            else:
                self.control.throttle = float(abs(action)) * 1 if action >= 0 else 0
                self.control.brake = float(abs(action)) * 1 if action < 0 else 0

        # Ego kinematics
        v_ego = self._calculate_velocity(self.ego_vehicle)
        v_front = self._calculate_velocity(self.front_vehicle)
        v_back = self._calculate_velocity(self.back_vehicle)
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x

        lane_center = self.client.get_world().get_map().get_waypoint(
            self.ego_vehicle.get_location(), project_to_road=True
        )

        ego_location = self.ego_vehicle.get_location()
        lane_center_location = lane_center.transform.location
        vector_to_lane_center = np.array(
            [lane_center_location.x - ego_location.x, lane_center_location.y - ego_location.y]
        )
        if self.y_ego_old is None:
            ego_forward_vector = np.array([1, 0])
        else:
            ego_forward_vector = np.array([x_ego - self.x_ego_old, y_ego - self.y_ego_old])

        self.y_ego_old = y_ego
        self.x_ego_old = x_ego

        lane_departure_dis = 0  # kept variable

        # Random braking routine
        if self.count >= self.brake_idx:
            self.brake_flag = 1

        marker = 0
        if self.brake_idx <= self.count < self.brake_idx + 80:
            if self.count == self.brake_idx:
                marker = 11
                self.sock.sendto(b"11", (self.UDP_IP, self.UDP_PORT))
                self.send_flag = 0
            elif self.control.brake > 0.5:
                marker = 31
                self.sock.sendto(b"31", (self.UDP_IP, self.UDP_PORT))
            self.front_control.brake = 0.5
            self.front_control.throttle = 0
            self.back_control.brake = 0.5
            self.back_control.throttle = 0
            self.front_vehicle.set_light_state(carla.VehicleLightState.Brake)

        if self.count == self.brake_idx + 80 and self.brake_flag == 1:
            self.sock.sendto(b"21", (self.UDP_IP, self.UDP_PORT))
            random_number = random.randint(200, 400)
            marker = 21
            self.brake_idx = random_number + self.brake_idx
            self.brake_flag = 0
            self.front_vehicle.set_light_state(carla.VehicleLightState.NONE)

        # Following behavior tuning
        dis = (
            (self.back_vehicle.get_location().x - self.front_vehicle.get_location().x) ** 2
            + (self.back_vehicle.get_location().y - self.front_vehicle.get_location().y) ** 2
        ) ** 0.5

        if dis > 60:
            self.back_control.throttle = self.front_control.throttle + 0.1
        if dis < 50:
            self.back_control.throttle = self.front_control.throttle - 0.1

        if self.town in (7, 4, 40):
            if dis > 50:
                self.back_control.throttle = self.front_control.throttle + 0.1
            if dis < 40:
                self.back_control.throttle = self.front_control.throttle - 0.1

        # Apply controls
        self.front_vehicle.apply_control(self.front_control)
        self.back_vehicle.apply_control(self.back_control)
        self.ego_vehicle.apply_control(self.control)

        # Log human intervention
        self.intervene_history.append(human_control)

        # Adopted action (human overrides)
        adopted_action = action if not self.intervention else human_control
        self.previous_action_list.append(adopted_action)

        # Travel distance accumulation
        dis_step = (
            (self.ego_vehicle.get_location().x - self.ego_location.x) ** 2
            + (self.ego_vehicle.get_location().y - self.ego_location.y) ** 2
        ) ** 0.5
        self.ego_location = self.ego_vehicle.get_location()
        self.travel_dis += dis_step

        # Control surrounding obs vehicles (if any)
        for index in range(len(self.obs_list)):
            obs_command = carla.VehicleControl()
            obs_command.steer = 0
            obs_velocity_diff = self.obs_velo_list[index] - abs(
                self.obs_list[index].get_velocity().x
            )
            obs_command.throttle = min(1, 0.4 * obs_velocity_diff) if obs_velocity_diff > 0 else 0
            obs_command.brake = min(1, -0.4 * obs_velocity_diff) if obs_velocity_diff < 0 else 0
            if index > 0:
                dis_to_front = abs(
                    self.obs_list[index - 1].get_location().x - self.obs_list[index].get_location().x
                )
                v_front_local = abs(self.obs_list[index - 1].get_velocity().x)
                v_current = abs(self.obs_list[index].get_velocity().x)
                if dis_to_front < 8 and v_current > v_front_local:
                    obs_command.brake = (8 - dis_to_front) / 4

            if self.conservative_surrounding:
                if 323 < y_ego < 328:
                    obs_command.throttle = 0
                    obs_command.brake = 0.2
            self.obs_list[index].apply_control(obs_command)

        # Next observation and distances
        next_observation, other_indicators = self.get_observation()
        dis_front = (
            (self.ego_vehicle.get_location().x - self.front_vehicle.get_location().x) ** 2
            + (self.ego_vehicle.get_location().y - self.front_vehicle.get_location().y) ** 2
        ) ** 0.5
        dis_ego = (
            (self.ego_vehicle.get_location().x - self.back_vehicle.get_location().x) ** 2
            + (self.ego_vehicle.get_location().y - self.back_vehicle.get_location().y) ** 2
        ) ** 0.5

        # Termination by collision / step count
        collision = self.get_collision_history()[1]
        collision = dis_front < 5 or dis_ego < 5
        if collision:
            self.collision = 1
        done = collision or self.count > 1000

        # Rewards
        punishment = min(dis_front, dis_ego) - 15
        t_gap = (dis_front - 4) / (v_ego + 0.0001)
        if t_gap < 1:
            punishment = max(-1 / t_gap, -10)
        if 1 <= t_gap < 2:
            punishment = t_gap
        if t_gap >= 2:
            punishment = max(-t_gap, -10)

        reward = -collision * 1000 + punishment + v_ego / 8

        if v_ego > v_front + 0.1:
            ttc = np.clip(dis_front / (v_ego - v_front + 0.0001), 0, 5) * 0.1
        else:
            ttc = 5

        print(reward)

        # Physical variables (diagnostics)
        acc_ego = (self.ego_vehicle.get_acceleration().x ** 2 + self.ego_vehicle.get_acceleration().y ** 2) ** 0.5
        acc_front = (self.front_vehicle.get_acceleration().x ** 2 + self.front_vehicle.get_acceleration().y ** 2) ** 0.5
        acc_back = (self.back_vehicle.get_acceleration().x ** 2 + self.back_vehicle.get_acceleration().y ** 2) ** 0.5
        reward = np.clip(reward, -1000, 1000)

        # Step counter
        self.count += 1

        # Diagnostics dict (unchanged keys)
        yaw_rate = (
            np.arctan(self.ego_vehicle.get_velocity().x / self.ego_vehicle.get_velocity().y)
            if self.ego_vehicle.get_velocity().y > 0
            else 0
        )
        physical_variables = {
            "velocity_y": self.ego_vehicle.get_velocity().y,
            "velocity_x": self.ego_vehicle.get_velocity().x,
            "position_y": self.ego_vehicle.get_location().y,
            "position_x": self.ego_vehicle.get_location().x,
            "yaw_rate": yaw_rate,
            "yaw": self.ego_vehicle.get_transform().rotation.yaw,
            "pitch": self.ego_vehicle.get_transform().rotation.pitch,
            "roll": self.ego_vehicle.get_transform().rotation.roll,
            "angular_velocity_y": self.ego_vehicle.get_angular_velocity().y,
            "angular_velocity_x": self.ego_vehicle.get_angular_velocity().x,
            "acceleration_x": self.ego_vehicle.get_acceleration().x,
            "acceleration_y": self.ego_vehicle.get_acceleration().y,
        }

        # Early termination via UI events
        if done or self.parse_events():
            self.terminate_position = y_ego
            self.post_process()

        return (
            next_observation,
            action,
            reward,
            self.intervention,
            done,
            physical_variables,
            dis_front,
            self.travel_dis,
            ttc,
            self.rgb,
        )

    # -------------------------------------------------------------------------
    # Teardown
    # -------------------------------------------------------------------------

    def destroy(self):
        """Destroy sensors and actors created by this environment."""
        if self.seman_camera is not None:
            self.seman_camera.stop()
        if self.viz_camera is not None:
            self.viz_camera.stop()
        if self.collision_sensor is not None:
            self.collision_sensor.stop()

        actors = [
            self.ego_vehicle,
            # self.vehicle_2,
            self.front_vehicle,
            self.back_vehicle,
            self.viz_camera,
            self.seman_camera,
            self.collision_sensor,
        ]
        actors.extend(self.obs_list)

        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])
        self.seman_camera = None
        self.viz_camera = None
        self.collision_sensor = None
        self.ego_vehicle = None

    def post_process(self):
        """Restore CARLA world settings and tear down actors."""
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
        if self.world is not None:
            self.destroy()

    def close(self):
        """Close pygame display."""
        pygame.display.quit()
        pygame.quit()

    def signal_handler(self, sig, frame):
        """SIGINT handler for clean exit."""
        print("Procedure terminated!")
        self.destroy()
        self.close()
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Observations / Events
    # -------------------------------------------------------------------------

    def get_observation(self):
        """Return resized semantic camera grayscale as state; other_indicators kept None."""
        state_space = self.camera_output[:, :, 0]
        state_space = cv2.resize(state_space, (WIDTH, HEIGHT))
        state_space = np.float16(np.squeeze(state_space) / 255)
        other_indicators = None
        return state_space, other_indicators

    def obtain_real_observation(self):
        """Return raw semantic camera channel (no resize/normalize)."""
        state_space = self.camera_output[:, :, 0]
        return state_space

    def parse_events(self):
        """Process pygame events; handle quit and HUD toggles."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    self._toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    self._next_weather(reverse=True)
                elif event.key == K_c:
                    self._next_weather()

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.intervention = False
                elif event.button == 1:
                    self._toggle_camera()
                elif event.button == 2:
                    self._next_sensor()

    # -------------------------------------------------------------------------
    # Human inputs (keyboard / wheel)
    # -------------------------------------------------------------------------

    def _parse_key(self):
        """Parse keyboard pedal input (W/S or Up/Down)."""
        keys = pygame.key.get_pressed()

        if (keys[K_UP] or keys[K_w]) and (not getattr(self, "w_pressed", 0)):
            self.human_default_throttle = 0.01
            self.w_pressed = 1
        elif (keys[K_UP] or keys[K_w]) and getattr(self, "w_pressed", 0):
            self.human_default_throttle += 0.05
        elif (not keys[K_UP]) and (not keys[K_w]):
            self.human_default_throttle = 0.0
            self.w_pressed = 0

        if (keys[K_DOWN] or keys[K_s]) and (not getattr(self, "s_pressed", 0)):
            self.human_default_brake = 0.01
            self.s_pressed = 1
        elif (keys[K_DOWN] or keys[K_s]) and getattr(self, "s_pressed", 0):
            self.human_default_brake += 0.1
        elif (not keys[K_DOWN]) and (not keys[K_s]):
            self.human_default_brake = 0.0
            self.s_pressed = 0

        if (keys[K_DOWN] or keys[K_s]) or (keys[K_UP] or keys[K_w]):
            human_throttle = np.clip(float(self.human_default_throttle), 0, 1)
            human_brake = np.clip(float(self.human_default_brake), 0, 1)
            human_action = human_throttle - human_brake
            self.keyboard_intervention = True
        else:
            human_action = None
            self.keyboard_intervention = False

        return human_action

    def _parse_wheel(self):
        """Parse joystick pedal inputs."""
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        if len(self.intervene_history) > 2:
            if abs(self.intervene_history[-2] - (jsInputs[self._throttle_idx] - jsInputs[self._brake_idx])) > 0.02:
                self.joystick_intervention = True
        if len(self.intervene_history) > 5:
            if abs(self.intervene_history[-5] - self.intervene_history[-1]) < 0.01:
                self.joystick_intervention = False

        if self.joystick_intervention:
            brakeSensitivity = 2.0
            throttleCmd = 1.6 + (
                2.05 * math.log10(-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2
            ) / (0.92 * brakeSensitivity)
            if throttleCmd <= 0:
                throttleCmd = 0
            elif throttleCmd > 1:
                throttleCmd = 1
            elif 0.62 < throttleCmd < 0.623:
                throttleCmd = 0

            brakeCmd = 1.6 + (
                2.05 * math.log10(-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2
            ) / 0.92
            if brakeCmd <= 0:
                brakeCmd = 0
            elif brakeCmd > 1:
                brakeCmd = 1
            elif 0.62 < brakeCmd < 0.623:
                brakeCmd = 0

            human_action = throttleCmd - brakeCmd
        else:
            human_action = None

        return human_action

    def _parse_key_steer(self):
        """Parse keyboard steering (A/D or Left/Right)."""
        keys = pygame.key.get_pressed()

        if (keys[K_LEFT] or keys[K_a]) and (not getattr(self, "a_pressed", 0)):
            self.human_default_steer = -0.02
            self.a_pressed = 1
        elif (keys[K_LEFT] or keys[K_a]) and getattr(self, "a_pressed", 0):
            self.human_default_steer -= 0.02
        elif (keys[K_RIGHT] or keys[K_d]) and (not getattr(self, "d_pressed", 0)):
            self.human_default_steer = 0.02
            self.d_pressed = 1
        elif (keys[K_RIGHT] or keys[K_d]) and getattr(self, "d_pressed", 0):
            self.human_default_steer += 0.02
        else:
            self.human_default_steer = 0
            self.d_pressed = 0
            self.a_pressed = 0

        if (keys[K_LEFT] or keys[K_a]) or (keys[K_RIGHT] or keys[K_d]):
            human_steer = np.clip(float(self.human_default_steer), -1, 1)
            human_action = human_steer
            self.keyboard_intervention = True
        else:
            human_action = None
            self.keyboard_intervention = False

        return human_action

    def _parse_wheel_steer(self):
        """Parse joystick steering."""
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]

        if len(self.intervene_history) > 2:
            if abs(self.intervene_history[-2] - (jsInputs[self._throttle_idx] - jsInputs[self._brake_idx])) > 0.02:
                self.joystick_intervention = True
        if len(self.intervene_history) > 5:
            if abs(self.intervene_history[-5] - self.intervene_history[-1]) < 0.01:
                self.joystick_intervention = False

        if self.joystick_intervention:
            human_steer = math.tan(1.1 * jsInputs[self._steer_idx])
            human_action = human_steer
        else:
            human_action = None

        return human_action

    # -------------------------------------------------------------------------
    # Spawners / Setup
    # -------------------------------------------------------------------------

    def _produce_vehicle_blueprint(self, x, y, yaw=0, color=0):
        """Produce a vehicle blueprint and a spawn point at (x, y, yaw)."""
        blueprint_library = self.world.get_blueprint_library()

        if self.vehicle_type != "single":
            bp = blueprint_library.filter("vehicle.*")[np.random.randint(0, 8)]
            while (
                int(bp.get_attribute("number_of_wheels")) != 4
                or bp.id == "vehicle.bmw.grandtourer"
                or bp.id == "vehicle.lincoln.mkz2017"
            ):
                bp = blueprint_library.filter("vehicle.*")[np.random.randint(0, 20)]
        else:
            bp = blueprint_library.filter("vehicle.audi.etron")[0]

        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z = 0.5
        spawn_point.rotation.yaw = yaw

        return bp, spawn_point

    def setup_sensors(self, vehicle, debug_mode=False):
        """
        Create sensors and attach them to the ego-vehicle.
        (Kept for parity; unchanged logic and signature.)
        """
        CarlaDataProvider = self.client
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        for sensor_spec in self.agent.sensors():
            if sensor_spec["type"].startswith("sensor.opendrive_map"):
                sensor = OpenDriveMapReader(vehicle, sensor_spec["reading_frequency"])  # noqa: F821
            elif sensor_spec["type"].startswith("sensor.speedometer"):
                delta_time = CarlaDataProvider.get_world().get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(vehicle, frame_rate)  # noqa: F821
            else:
                bp = bp_library.find(str(sensor_spec["type"]))
                if sensor_spec["type"].startswith("sensor.camera.semantic_segmentation"):
                    bp.set_attribute("image_size_x", str(sensor_spec["width"]))
                    bp.set_attribute("image_size_y", str(sensor_spec["height"]))
                    bp.set_attribute("fov", str(sensor_spec["fov"]))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                elif sensor_spec["type"].startswith("sensor.camera.depth"):
                    bp.set_attribute("image_size_x", str(sensor_spec["width"]))
                    bp.set_attribute("image_size_y", str(sensor_spec["height"]))
                    bp.set_attribute("fov", str(sensor_spec["fov"]))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                elif sensor_spec["type"].startswith("sensor.camera"):
                    bp.set_attribute("image_size_x", str(sensor_spec["width"]))
                    bp.set_attribute("image_size_y", str(sensor_spec["height"]))
                    bp.set_attribute("fov", str(sensor_spec["fov"]))
                    bp.set_attribute("lens_circle_multiplier", str(3.0))
                    bp.set_attribute("lens_circle_falloff", str(3.0))
                    bp.set_attribute("chromatic_aberration_intensity", str(0.5))
                    bp.set_attribute("chromatic_aberration_offset", str(0))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                elif sensor_spec["type"].startswith("sensor.lidar.ray_cast_semantic"):
                    bp.set_attribute("range", str(85))
                    bp.set_attribute("rotation_frequency", str(10))
                    bp.set_attribute("channels", str(64))
                    bp.set_attribute("upper_fov", str(10))
                    bp.set_attribute("lower_fov", str(-30))
                    bp.set_attribute("points_per_second", str(600000))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                elif sensor_spec["type"].startswith("sensor.lidar"):
                    bp.set_attribute("range", str(85))
                    bp.set_attribute("rotation_frequency", str(10))
                    bp.set_attribute("channels", str(64))
                    bp.set_attribute("upper_fov", str(10))
                    bp.set_attribute("lower_fov", str(-30))
                    bp.set_attribute("points_per_second", str(600000))
                    bp.set_attribute("atmosphere_attenuation_rate", str(0.004))
                    bp.set_attribute("dropoff_general_rate", str(0.45))
                    bp.set_attribute("dropoff_intensity_limit", str(0.8))
                    bp.set_attribute("dropoff_zero_intensity", str(0.4))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                elif sensor_spec["type"].startswith("sensor.other.radar"):
                    bp.set_attribute("horizontal_fov", str(sensor_spec["fov"]))
                    bp.set_attribute("vertical_fov", str(sensor_spec["fov"]))
                    bp.set_attribute("points_per_second", "1500")
                    bp.set_attribute("range", "100")
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                elif sensor_spec["type"].startswith("sensor.other.gnss"):
                    bp.set_attribute("noise_alt_bias", str(0.0))
                    bp.set_attribute("noise_lat_bias", str(0.0))
                    bp.set_attribute("noise_lon_bias", str(0.0))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation()
                elif sensor_spec["type"].startswith("sensor.other.imu"):
                    bp.set_attribute("noise_accel_stddev_x", str(0.001))
                    bp.set_attribute("noise_accel_stddev_y", str(0.001))
                    bp.set_attribute("noise_accel_stddev_z", str(0.015))
                    bp.set_attribute("noise_gyro_stddev_x", str(0.001))
                    bp.set_attribute("noise_gyro_stddev_y", str(0.001))
                    bp.set_attribute("noise_gyro_stddev_z", str(0.001))
                    sensor_location = carla.Location(
                        x=sensor_spec["x"], y=sensor_spec["y"], z=sensor_spec["z"]
                    )
                    sensor_rotation = carla.Rotation(
                        pitch=sensor_spec["pitch"],
                        roll=sensor_spec["roll"],
                        yaw=sensor_spec["yaw"],
                    )
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = CarlaDataProvider.get_world().spawn_actor(
                    bp, sensor_transform, vehicle
                )

        CarlaDataProvider.get_world().tick()

    def _produce_walker_blueprint(self, x, y):
        """Produce a walker blueprint and spawn point."""
        bp = self.world.get_blueprint_library().filter("walker.*")[np.random.randint(2)]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_point.location.x = x
        spawn_point.location.y = y
        spawn_point.location.z += 0.1
        spawn_point.rotation.yaw = 0
        return bp, spawn_point

    # -------------------------------------------------------------------------
    # UI helpers / Kinematics
    # -------------------------------------------------------------------------

    def _toggle_camera(self):
        """Cycle through predefined camera transforms."""
        self.camera_transform_index = (self.camera_transform_index + 1) % len(self.camera_transforms)

    def _next_sensor(self):
        """Cycle to next sensor (placeholder)."""
        self.camera_index += 1  # noqa: B018 (kept as-is)

    def _next_weather(self, reverse=False):
        """Cycle to next weather preset."""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])

    def _calculate_velocity(self, actor):
        """Return Euclidean velocity magnitude for a CARLA actor."""
        return (actor.get_velocity().x ** 2 + actor.get_velocity().y ** 2 + actor.get_velocity().z ** 2) ** 0.5

    def _dis_p_to_l(self, k, b, x, y):
        """Distance from point to line (k,b) with sigmoid shaping."""
        dis = abs((k * x - y + b) / math.sqrt(k * k + 1))
        return self._sigmoid(dis, 2)

    def _calculate_k_b(self, x1, y1, x2, y2):
        """Return slope k and intercept b of the line through (x1,y1),(x2,y2)."""
        k = (y1 - y2) / (x1 - x2)
        b = (x1 * y2 - x2 * y1) / (x1 - x2)
        return k, b

    def _dis_p_to_p(self, x1, y1, x2, y2):
        """Point-to-point Euclidean distance."""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _to_corner_coordinate(self, x, y, yaw):
        """Return 4 corners of a rectangle given center (x,y) and yaw."""
        xa = x + 2.64 * math.cos(yaw * math.pi / 180 - 0.43)
        ya = y + 2.64 * math.sin(yaw * math.pi / 180 - 0.43)
        xb = x + 2.64 * math.cos(yaw * math.pi / 180 + 0.43)
        yb = y + 2.64 * math.cos(yaw * math.pi / 180 + 0.43)
        xc = x + 2.64 * math.cos(yaw * math.pi / 180 - 0.43 + math.pi)
        yc = y + 2.64 * math.cos(yaw * math.pi / 180 - 0.43 + math.pi)
        xd = x + 2.64 * math.cos(yaw * math.pi / 180 + 0.43 + math.pi)
        yd = y + 2.64 * math.cos(yaw * math.pi / 180 + 0.43 + math.pi)
        return xa, ya, xb, yb, xc, yc, xd, yd

    def _sigmoid(self, x, theta):
        """Scaled sigmoid in [-1,1]."""
        return 2.0 / (1 + math.exp(-theta * x)) - 1

    @staticmethod
    def _is_quit_shortcut(key):
        """Keyboard shortcut to quit (Esc or Ctrl+Q)."""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# -----------------------------------------------------------------------------
# Weather presets
# -----------------------------------------------------------------------------
def find_weather_presets():
    """Return list of (WeatherParameters, name) tuples available in CARLA."""
    rgx = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))  # noqa: E731
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
