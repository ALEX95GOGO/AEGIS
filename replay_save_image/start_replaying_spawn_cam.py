import glob
import os
import sys
import time
import math
import weakref
from DReyeVR_utils import find_ego_vehicle
import queue
import re
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random
import subprocess
import ctypes
import sys
import psutil

def kill_process(process_name):
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == process_name:
            pid = process.info['pid']
            try:
                # Terminate the process
                psutil.Process(pid).terminate()
                print(f"Process {process_name} (PID: {pid}) terminated.")
            except Exception as e:
                print(f"Error terminating process {process_name}: {e}")



def run_as_admin():
    if sys.platform == 'win32':
        try:
            # Use ctypes to call ShellExecute with "runas" verb
            ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",
                sys.executable,  # The executable to run (in this case, the Python interpreter)
                " ".join(sys.argv),  # The arguments to pass to the executable
                None,
                1  # Show the window
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("This script requires Windows to run with administrative privileges.")
        sys.exit(1)

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def find_bin_file(path, condition):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".bin") and all(keyword in file for keyword in condition):
                return os.path.join(root, file)

    return None

def main(condition='test', sub_id='test'):
    client = carla.Client('127.0.0.1', 9020)
    client.set_timeout(60.0)

    try:

        world = client.get_world() 
        ego_vehicle = None
        ego_cam = None
        depth_cam = None
        depth_cam02 = None
        sem_cam = None
        rad_ego = None
        lidar_sen = None

        # --------------
        # Query the recording
        # --------------
        """
        # Show the most important events in the recording.  
        print(client.show_recorder_file_info("~/tutorial/recorder/recording05.log",False))
        # Show actors not moving 1 meter in 10 seconds.  
        #print(client.show_recorder_actors_blocked("~/tutorial/recorder/recording04.log",10,1))
        # Show collisions between any type of actor.  
        #print(client.show_recorder_collisions("~/tutorial/recorder/recording04.log",'v','a'))
        """

        # --------------
        # Reenact a fragment of the recording
        # --------------
        #replay_file(self, name, start, duration, follow_id, replay_sensors)
        
        # set the time factor for the replayer
        #client.set_replayer_time_factor(0.5)

        # set to ignore the hero vehicles or not
        #client.set_replayer_ignore_hero(True)
        #client.set_replayer_ignore_spectator(True)

        #condition= 'steer_night'

        # Specify the path to search
        search_path = r"D:\VR_driving_data\eye-tracking\{}".format(sub_id)
        # Call the function to find the .bin file
        #import pdb; pdb.set_trace()
        result = find_bin_file(search_path, condition)
        print(result)
        client.replay_file(result,0,0,0)
        world.tick()
        client.set_replayer_ignore_hero(True)

        # --------------
        # Set playback simulation conditions
        # --------------
        #201 ego vehicle
        ego_vehicle = world.get_actor(201) #Store the ID from the simulation or query the recording to find out
        #ego_vehicle = world.get_blueprint_library().filter('vehicle.audi.etron')[0]
        #bp_ego.set_attribute('role_name', 'hero')
        #ego_vehicle.destroy()
        #ego_vehicle = find_ego_vehicle(world)
        # --------------
        # Place spectator on ego spawning
        # --------------
        spectator = world.get_spectator()
        #import pdb; pdb.set_trace()
        world_snapshot = world.wait_for_tick() 
        
        
        #hero_bp[0].set_attribute('role_name', 'hero')
        # --------------
        # Change weather conditions
        # --------------
        #import pdb; pdb.set_trace()
        if condition[-3:] == 'day':
            _weather_presets = find_weather_presets()
            preset = _weather_presets[2]
            #import pdb;pdb.set_trace()
            world.set_weather(preset[0])
        else:
            _weather_presets = find_weather_presets()
            preset = _weather_presets[0]
            #import pdb;pdb.set_trace()
            world.set_weather(preset[0])
        """
        weather = world.get_weather()
        weather.sun_altitude_angle = -30
        weather.fog_density = 65
        weather.fog_distance = 10
        world.set_weather(weather)
        """

        # --------------
        # Add a RGB camera to ego vehicle.
        # --------------
        
        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_location = carla.Location(1,0,2)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        cam_bp.set_attribute('sensor_tick', '0.0')
        cam_bp.set_attribute("image_size_x",str(1920))
        cam_bp.set_attribute("image_size_y",str(1080))
        #cam_bp.set_attribute("image_size_x",str(1920/5))
        #cam_bp.set_attribute("image_size_y",str(1080/5))
        cam_bp.set_attribute("fov",str(105))
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        #ego_cam.listen(lambda image: image.save_to_disk('~/tutorial/new_rgb_output/%.6d.jpg' % image.frame))
        
        image_queue = queue.Queue()
        ego_cam.listen(image_queue.put)

        spectator.set_transform(cam_transform)
        # --------------
        # Add a Logarithmic Depth camera to ego vehicle. 
        # --------------
        
        depth_cam = None
        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x",str(1920/5))
        depth_bp.set_attribute("image_size_y",str(1080/5))
        depth_bp.set_attribute("fov",str(105))
        depth_bp.set_attribute('sensor_tick', '0.0')
        depth_location = carla.Location(1,0,2)
        depth_rotation = carla.Rotation(0,0,0)
        depth_transform = carla.Transform(depth_location,depth_rotation)
        depth_cam = world.spawn_actor(depth_bp,depth_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        #depth_cam.listen(lambda image: image.save_to_disk('~/tutorial/de_log/%.6d.jpg' % image.frame,carla.ColorConverter.LogarithmicDepth))
        depth_queue = queue.Queue()
        depth_cam.listen(depth_queue.put)
        # --------------
        # Add a Depth camera to ego vehicle. 
        # --------------
        
        depth_cam02 = None
        depth_bp02 = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp02.set_attribute("image_size_x",str(1920))
        depth_bp02.set_attribute("image_size_y",str(1080))
        depth_bp02.set_attribute("fov",str(105))
        depth_location02 = carla.Location(2,0,1)
        depth_rotation02 = carla.Rotation(0,180,0)
        depth_transform02 = carla.Transform(depth_location02,depth_rotation02)
        depth_cam02 = world.spawn_actor(depth_bp02,depth_transform02,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        depth_cam02.listen(lambda image: image.save_to_disk('~/tutorial/de/%.6d.jpg' % image.frame,carla.ColorConverter.Depth))
        

        # --------------
        # Add a new semantic segmentation camera to ego vehicle
        # --------------
        
        sem_cam = None
        sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        #sem_bp.set_attribute("image_size_x",str(1920))
        #sem_bp.set_attribute("image_size_y",str(1080))
        sem_bp.set_attribute("image_size_x",str(1920/5))
        sem_bp.set_attribute("image_size_y",str(1080/5))
        sem_bp.set_attribute("fov",str(105))
        sem_bp.set_attribute('sensor_tick', '0.0')
        sem_location = carla.Location(1,0,2)
        sem_rotation = carla.Rotation(0,0,0)
        sem_transform = carla.Transform(sem_location,sem_rotation)
        sem_cam = world.spawn_actor(sem_bp,sem_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        #sem_cam.listen(lambda image: image.save_to_disk('~/tutorial/new_sem_output/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette))
        sem_queue = queue.Queue()
        sem_cam.listen(sem_queue.put)
        
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.025
        #settings.fixed_delta_seconds = None
        settings.synchronous_mode = True # Enables synchronous mode
        world.apply_settings(settings)
        
        # --------------
        # Add a new radar sensor to ego vehicle
        # --------------
        """
        rad_cam = None
        rad_bp = world.get_blueprint_library().find('sensor.other.radar')
        rad_bp.set_attribute('horizontal_fov', str(35))
        rad_bp.set_attribute('vertical_fov', str(20))
        rad_bp.set_attribute('range', str(20))
        rad_location = carla.Location(x=2.8, z=1.0)
        rad_rotation = carla.Rotation(pitch=5)
        rad_transform = carla.Transform(rad_location,rad_rotation)
        rad_ego = world.spawn_actor(rad_bp,rad_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
        def rad_callback(radar_data):
            velocity_range = 7.5 # m/s
            current_rot = radar_data.transform.rotation
            for detect in radar_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                # The 0.25 adjusts a bit the distance so the dots can
                # be properly seen
                fw_vec = carla.Vector3D(x=detect.depth - 0.25)
                carla.Transform(
                    carla.Location(),
                    carla.Rotation(
                        pitch=current_rot.pitch + alt,
                        yaw=current_rot.yaw + azi,
                        roll=current_rot.roll)).transform(fw_vec)

                def clamp(min_v, max_v, value):
                    return max(min_v, min(value, max_v))

                norm_velocity = detect.velocity / velocity_range # range [-1, 1]
                r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
                g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
                b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
                world.debug.draw_point(
                    radar_data.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.06,
                    persistent_lines=False,
                    color=carla.Color(r, g, b))
        rad_ego.listen(lambda radar_data: rad_callback(radar_data))
        """

        # --------------
        # Add a new LIDAR sensor to ego vehicle
        # --------------
        """
        lidar_cam = None
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(32))
        lidar_bp.set_attribute('points_per_second',str(90000))
        lidar_bp.set_attribute('rotation_frequency',str(40))
        lidar_bp.set_attribute('range',str(20))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk('/home/adas/Desktop/tutorial/new_lidar_output/%.6d.ply' % point_cloud.frame))
        """

        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            try:
                world_snapshot = world.tick()
                spectator.set_transform(carla.Transform(cam_location,cam_rotation))
                
                image = image_queue.get()
                image.save_to_disk('{}/{}/new_rgb_output/%.6d.jpg'.format(sub_id, condition) % image.frame)

                sem = sem_queue.get()
                print(sem.frame)
                sem.save_to_disk('{}/{}/new_sem_output/%.6d.jpg'.format(sub_id, condition) % sem.frame, carla.ColorConverter.CityScapesPalette)
                if sem.frame>12000:
                    break

                depth = depth_queue.get()
                depth.save_to_disk('{}/{}/new_depth_output/%.6d.jpg'.format(sub_id, condition) % depth.frame,carla.ColorConverter.LogarithmicDepth)
                
                
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                break
            #print(_parse_image(cameras[1][1],image))
            #world_snapshot = world.wait_for_tick()
            #print(depth_cam.shape)
            #time.sleep(1)
            #spectator.set_transform(ego_vehicle.get_transform())
    
    finally:
        # --------------
        # Destroy actors
        # --------------
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            if depth_cam is not None:
                depth_cam.stop()
                depth_cam.destroy()
            if sem_cam is not None:
                sem_cam.stop()
                sem_cam.destroy()
            if rad_ego is not None:
                rad_ego.stop()
                rad_ego.destroy()
            if lidar_sen is not None:
                lidar_sen.stop()
                lidar_sen.destroy()
            ego_vehicle.destroy()
        print('\nNothing to be done.')

def _parse_image(bp, image):

    image.convert(bp)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.array(image.raw_data)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    
    return array

if __name__ == '__main__':
    #'manual_day', 
    condition = [ 'pedal_day', 'steer_day', 'manual_night', 'pedal_night', 'steer_night','manual_day']
    #condition = ['pedal_day_4','pedal_day_7', 'pedal_night_4','pedal_night_7']
    #sub_id = ['S11','S12','S13','S14']
    #sub_id = ['S17','S18']
    sub_id = ['S09']

    #sub_id = ['S30']
    #condition = ['DOS1','DOS2','DOS3','DOS4']

    for id in range(len(sub_id)):
        for i in range(len(condition)):

            # Specify the process name you want to terminate
            process_name = "CarlaUE4-Win64-Shipping.exe"

            # Call the function to terminate the process
            kill_process(process_name)

            time.sleep(5)

            # Check if the script is running with administrative privileges
            #subprocess.run([r"D:\CARLA_9_13\carla\Build\UE4Carla-v3\0.9.13-dirty\WindowsNoEditor\CarlaUE4.exe"])
            carla_process = subprocess.Popen([r"D:\CARLA_9_13\carla\Build\UE4Carla-v3\0.9.13-dirty\WindowsNoEditor\CarlaUE4.exe", "-carla-port=9020"])

            #if ctypes.windll.shell32.IsUserAnAdmin() == 0:
                # If not, run the script as administrator
            #    run_as_admin()
            #else:
                # If already running as administrator, perform the desired actions
                # For example, you can run your .exe file here
            #    subprocess.run([r"D:\CARLA_9_13\carla\Build\UE4Carla-v3\0.9.13-dirty\WindowsNoEditor\CarlaUE4.exe"])
            # Specify the current and new file names
            
            try:
                #import pdb; pdb.set_trace()
                
                print(condition[i])
                time.sleep(10)
                main(condition=condition[i], sub_id=sub_id[id])
            except KeyboardInterrupt:
                pass
                
            current_filename = r"C:\Users\11550\Documents\carla\ViewSize.txt"

            #if i >= 1:
            new_filename = r"C:\Users\11550\Documents\carla\ViewSize_{}_{}.txt".format(sub_id[id], condition[i])

            # Assuming the file is in the current working directory, you can specify the full path if needed
            current_filepath = os.path.join(os.getcwd(), current_filename)
            new_filepath = os.path.join(os.getcwd(), new_filename)

            try:
                # Rename the file
                os.rename(current_filepath, new_filepath)
                print(f"File '{current_filename}' has been renamed to '{new_filename}'.")
            except FileNotFoundError:
                print(f"File '{current_filename}' not found.")
            except FileExistsError:
                print(f"File '{new_filename}' already exists.")
            except Exception as e:
                print(f"An error occurred: {e}")

            finally:
                print('\nDone with tutorial_replay.')