#!/usr/bin/env python3
"""
Replay a single CARLA recorder .bin file and capture sensor outputs.

Usage (Windows example):
  python replay_single_bin.py --bin-file "D:\VR_driving_data\eye-tracking\S09\recording.bin" --out-dir "D:\output"

Notes:
- This script assumes the replayed recording spawns the ego vehicle with actor id 201.
  If that’s not true for your recordings, use --ego-id to change it.
- Runs in synchronous mode and saves RGB / semantic / depth images.
"""

import glob
import os
import sys
import time
import math
import queue
import re
import argparse
import logging
import subprocess
import ctypes

import psutil

# Try to locate CARLA egg
try:
    sys.path.append(glob.glob(
        r'../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        )
    )[0])
except IndexError:
    pass

import carla  # noqa: E402


def kill_process(process_name: str) -> None:
    for process in psutil.process_iter(['pid', 'name']):
        if process.info.get('name') == process_name:
            pid = process.info['pid']
            try:
                psutil.Process(pid).terminate()
                print(f"Process {process_name} (PID: {pid}) terminated.")
            except Exception as e:
                print(f"Error terminating process {process_name}: {e}")


def run_as_admin() -> None:
    """Optional helper if you need admin privileges on Windows."""
    if sys.platform == 'win32':
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,
                " ".join(sys.argv),
                None,
                1
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("This script requires Windows to run with administrative privileges.")
        sys.exit(1)


def find_weather_presets():
    rgx = re.compile(r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match(r'[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def resolve_bin_file(bin_file: str) -> str:
    if not bin_file:
        raise ValueError("You must provide --bin-file.")
    bin_file = os.path.expandvars(os.path.expanduser(bin_file))
    bin_file = os.path.abspath(bin_file)
    if not os.path.isfile(bin_file):
        raise FileNotFoundError(f"BIN file not found: {bin_file}")
    if not bin_file.lower().endswith(".bin"):
        raise ValueError(f"Not a .bin file: {bin_file}")
    return bin_file


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def spawn_sensors(world: carla.World, ego_vehicle: carla.Actor,
                  rgb_size=(1920, 1080), small_scale=5, fov=105.0):
    """
    Spawns:
      - RGB camera (full res)
      - Log depth camera (downscaled)
      - Semantic segmentation camera (downscaled)
      - Optional second depth camera (full res) that saves directly (kept from your original)
    Returns: (actors_dict, queues_dict)
    """
    bp_lib = world.get_blueprint_library()

    actors = {}
    queues = {}

    # RGB camera (full res)
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('sensor_tick', '0.0')
    cam_bp.set_attribute("image_size_x", str(int(rgb_size[0])))
    cam_bp.set_attribute("image_size_y", str(int(rgb_size[1])))
    cam_bp.set_attribute("fov", str(float(fov)))

    cam_transform = carla.Transform(carla.Location(1, 0, 2), carla.Rotation(0, 0, 0))
    rgb_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle,
                                attachment_type=carla.AttachmentType.Rigid)
    rgb_q = queue.Queue()
    rgb_cam.listen(rgb_q.put)

    actors["rgb_cam"] = rgb_cam
    queues["rgb"] = rgb_q

    # Log depth (downscaled)
    depth_bp = bp_lib.find('sensor.camera.depth')
    depth_bp.set_attribute('sensor_tick', '0.0')
    depth_bp.set_attribute("image_size_x", str(int(rgb_size[0] / small_scale)))
    depth_bp.set_attribute("image_size_y", str(int(rgb_size[1] / small_scale)))
    depth_bp.set_attribute("fov", str(float(fov)))

    depth_transform = carla.Transform(carla.Location(1, 0, 2), carla.Rotation(0, 0, 0))
    depth_cam = world.spawn_actor(depth_bp, depth_transform, attach_to=ego_vehicle,
                                  attachment_type=carla.AttachmentType.Rigid)
    depth_q = queue.Queue()
    depth_cam.listen(depth_q.put)

    actors["depth_cam"] = depth_cam
    queues["depth"] = depth_q

    # Semantic (downscaled)
    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('sensor_tick', '0.0')
    sem_bp.set_attribute("image_size_x", str(int(rgb_size[0] / small_scale)))
    sem_bp.set_attribute("image_size_y", str(int(rgb_size[1] / small_scale)))
    sem_bp.set_attribute("fov", str(float(fov)))

    sem_transform = carla.Transform(carla.Location(1, 0, 2), carla.Rotation(0, 0, 0))
    sem_cam = world.spawn_actor(sem_bp, sem_transform, attach_to=ego_vehicle,
                                attachment_type=carla.AttachmentType.Rigid)
    sem_q = queue.Queue()
    sem_cam.listen(sem_q.put)

    actors["sem_cam"] = sem_cam
    queues["sem"] = sem_q

    # Extra depth camera (full res) saves directly (optional)
    depth_bp02 = bp_lib.find('sensor.camera.depth')
    depth_bp02.set_attribute("image_size_x", str(int(rgb_size[0])))
    depth_bp02.set_attribute("image_size_y", str(int(rgb_size[1])))
    depth_bp02.set_attribute("fov", str(float(fov)))

    depth_transform02 = carla.Transform(carla.Location(2, 0, 1), carla.Rotation(0, 180, 0))
    depth_cam02 = world.spawn_actor(depth_bp02, depth_transform02, attach_to=ego_vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)

    actors["depth_cam02"] = depth_cam02

    return actors, queues, cam_transform


def set_sync(world: carla.World, fixed_dt: float) -> None:
    settings = world.get_settings()
    settings.fixed_delta_seconds = fixed_dt
    settings.synchronous_mode = True
    world.apply_settings(settings)


def restore_async(world: carla.World) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


def replay_single_bin(bin_file: str,
                      host: str,
                      port: int,
                      out_dir: str,
                      ego_id: int,
                      fixed_dt: float,
                      max_frames: int,
                      set_weather: str | None):
    """
    Replays the recorder file and saves sensor outputs.
    """
    bin_path = resolve_bin_file(bin_file)
    ensure_dir(out_dir)

    rgb_dir = os.path.join(out_dir, "rgb")
    sem_dir = os.path.join(out_dir, "sem")
    depth_dir = os.path.join(out_dir, "depth")
    depth02_dir = os.path.join(out_dir, "depth02")
    for d in (rgb_dir, sem_dir, depth_dir, depth02_dir):
        ensure_dir(d)

    client = carla.Client(host, port)
    client.set_timeout(60.0)

    world = client.get_world()

    actors = {}
    queues = {}

    try:
        print(f"Replaying: {bin_path}")
        client.replay_file(bin_path, 0, 0, 0)
        world.tick()

        # Ignore hero and spectator from replay if you want (kept from your original)
        client.set_replayer_ignore_hero(True)

        # Optional weather override
        if set_weather:
            presets = dict(find_weather_presets())
            # allow matching by "Clear Noon" name (case-insensitive)
            match = None
            for wp, name in find_weather_presets():
                if name.lower() == set_weather.lower():
                    match = wp
                    break
            if match is None:
                raise ValueError(f"Unknown weather preset name '{set_weather}'. "
                                 f"Run with --list-weather to see options.")
            world.set_weather(match)

        # Get ego vehicle
        ego_vehicle = world.get_actor(ego_id)
        if ego_vehicle is None:
            raise RuntimeError(
                f"Could not find ego vehicle with id={ego_id}. "
                f"Try a different --ego-id."
            )

        # Place spectator
        spectator = world.get_spectator()

        # Spawn sensors
        actors, queues, cam_transform = spawn_sensors(world, ego_vehicle)
        spectator.set_transform(cam_transform)

        # Set sync mode AFTER sensors created (works either way, but consistent)
        set_sync(world, fixed_dt=fixed_dt)

        # Depth cam02 saves directly (kept behavior)
        actors["depth_cam02"].listen(
            lambda image: image.save_to_disk(
                os.path.join(depth02_dir, '%.6d.jpg' % image.frame),
                carla.ColorConverter.Depth
            )
        )

        # Main loop
        frames = 0
        while True:
            world_snapshot = world.tick()
            spectator.set_transform(cam_transform)

            # RGB
            rgb = queues["rgb"].get()
            rgb.save_to_disk(os.path.join(rgb_dir, '%.6d.jpg' % rgb.frame))

            # Semantic
            sem = queues["sem"].get()
            sem.save_to_disk(
                os.path.join(sem_dir, '%.6d.jpg' % sem.frame),
                carla.ColorConverter.CityScapesPalette
            )

            # Log depth
            depth = queues["depth"].get()
            depth.save_to_disk(
                os.path.join(depth_dir, '%.6d.jpg' % depth.frame),
                carla.ColorConverter.LogarithmicDepth
            )

            frames += 1
            if max_frames > 0 and frames >= max_frames:
                print(f"Reached max_frames={max_frames}. Stopping.")
                break

    except Exception as e:
        logging.exception(f"Replay failed: {e}")
        raise
    finally:
        # Clean up sensors
        try:
            for k, a in actors.items():
                try:
                    a.stop()
                except Exception:
                    pass
                try:
                    a.destroy()
                except Exception:
                    pass
        finally:
            # Restore async mode
            try:
                restore_async(world)
            except Exception:
                pass
        print("Done.")


def list_weather_and_exit():
    presets = find_weather_presets()
    for _, name in presets:
        print(name)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-file", required=True, help="Path to a single .bin recorder file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9020)
    parser.add_argument("--out-dir", default=os.path.abspath("./replay_output"))
    parser.add_argument("--ego-id", type=int, default=201, help="Ego actor id in the replay")
    parser.add_argument("--fixed-dt", type=float, default=0.025)
    parser.add_argument("--max-frames", type=int, default=12000, help="0 = unlimited")
    parser.add_argument("--weather", default=None, help="Optional weather preset name (e.g. 'Clear Noon')")
    parser.add_argument("--list-weather", action="store_true", help="Print weather preset names and exit")
    parser.add_argument("--kill-server", action="store_true", help="Kill CarlaUE4-Win64-Shipping.exe before running")
    parser.add_argument("--carla-exe", default=None,
                        help="Optional path to CarlaUE4.exe to launch automatically. If omitted, script assumes CARLA is already running.")
    parser.add_argument("--carla-port-arg", default=None,
                        help="Optional, e.g. '-carla-port=9020'. If omitted and --carla-exe is set, defaults to port.")
    args = parser.parse_args()

    if args.list_weather:
        list_weather_and_exit()

    if args.kill_server:
        kill_process("CarlaUE4-Win64-Shipping.exe")
        time.sleep(3)

    # Optionally launch CARLA server
    carla_proc = None
    if args.carla_exe:
        port_arg = args.carla_port_arg or f"-carla-port={args.port}"
        carla_proc = subprocess.Popen([args.carla_exe, port_arg])
        # Give server a moment to boot
        time.sleep(10)

    try:
        replay_single_bin(
            bin_file=args.bin_file,
            host=args.host,
            port=args.port,
            out_dir=args.out_dir,
            ego_id=args.ego_id,
            fixed_dt=args.fixed_dt,
            max_frames=args.max_frames,
            set_weather=args.weather
        )
    finally:
        if carla_proc is not None:
            try:
                carla_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
