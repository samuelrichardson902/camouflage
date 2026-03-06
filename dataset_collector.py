import numpy as np
import cv2
import time
import random
import os
import sys
from tqdm import tqdm
from CarlaHandler import *


output_root = "./sample_dataset"

vehicle_ids = [
    'vehicle.tesla.model3', 
    'vehicle.audi.tt', 
    'vehicle.toyota.prius', 
    'vehicle.seat.leon', 
    'vehicle.nissan.patrol'
]

distances= [5,7,10] # camera distances
pitches= [10,30,45] # camera height (degrees)
yaw_step =30 # degrees between shots 360/30=12 shots
locations_per_car = 45 # how many different spawns to test each car


town_id = 'Town03'
res = 500
car_colour = (124, 124, 124)  # grey reference
diff_colour = (124, 14, 14)   # for differential mask


def ensure_folders():
    subfolders = ['reference', 'masks', 'overlays', 'transforms']
    for folder in subfolders:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)


def randomize_lighting(handler):
    # randomize lighting for every position
    handler.update_sun_azimuth_angle(random.randint(0, 360))
    handler.update_sun_altitude_angle(random.randint(15, 90))
    
    
    # clear weather to ensure masks are clean
    handler.update_wind_intensity(0.0)
    handler.update_cloudiness(0.0)
    handler.update_precipitation(0.0)
    
    handler.world_tick(5)


def set_camera(handler, distance, pitch, yaw):
    # to update camera pose
    handler.update_distance(distance)
    handler.update_pitch(pitch)
    handler.update_yaw(yaw)
    handler.world_tick(5)
    time.sleep(0.02)
    
    


def process_car_mask(mask, min_pixels=700):
    # clean & only take largest blob which should be the car
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2: return False, mask 

    # sort by area, ignore index 0 (background)
    component_indices = np.argsort(stats[1:, 4])[::-1] + 1
    largest_label = component_indices[0]
    
    if stats[largest_label, 4] < min_pixels: return False, mask
    
    cleaned_mask = (labels == largest_label).astype(np.uint8) * 255
    return True, cleaned_mask


def capture_sample(handler, save_idx, meta_data):
    # performs the Reference -> Mask -> Color Change -> Diff workflow.

    # ref image
    handler.change_vehicle_color(car_colour)
    handler.world_tick(20) 
    time.sleep(0.05)
    ref_image = handler.get_image()
    
    # segmentation mask
    seg_image = handler.get_segmentation()
    raw_mask = (seg_image[:,:,0] == 255) & (seg_image[:,:,1] == 0) & (seg_image[:,:,2] == 0)
    is_valid, vehicle_mask = process_car_mask(raw_mask)
    if not is_valid: return False

    # diff img
    handler.change_vehicle_color(diff_colour)
    handler.world_tick(20)
    time.sleep(0.05)
    cross_ref_image = handler.get_image()

    # get feature mask based on what pixels did/didnt change
    
    intersection = np.isclose(ref_image, cross_ref_image, atol=6).all(axis=-1)
    
    # get unchanged pixels within the car mask
    feature_mask = np.zeros_like(intersection)
    feature_mask[vehicle_mask > 0] = intersection[vehicle_mask > 0]
    
    feature_overlay = np.where(feature_mask[:, :, np.newaxis], ref_image, 0)

    # save data
    filename = f"{save_idx:05d}.png"
    cv2.imwrite(f"{output_root}/reference/{filename}", ref_image)
    cv2.imwrite(f"{output_root}/masks/{filename}", vehicle_mask)
    cv2.imwrite(f"{output_root}/overlays/{filename}", feature_overlay)
    
    # save [distance, pitch, yaw, vehicle_id_string]
    np.save(f"{output_root}/transforms/{save_idx:05d}.npy", np.array(meta_data))
    
    return True

def main():
    ensure_folders()

    shots_per_orbit = len(range(0,360,yaw_step))
    total_samples = len(vehicle_ids) * locations_per_car * len(distances) * len(pitches) * shots_per_orbit
    print(f"{total_samples} images to be collected")


    # init carla
    handler = CarlaHandler(x_res=res, y_res=res, town=town_id)
    time.sleep(2)
    handler.world_tick(10)
    n_spawn_points = handler.get_spawn_points()

    global_counter = 0

    with tqdm(total=total_samples, desc="Collecting Samples", unit="img") as pbar:

        # iterate over different vehicles
        for v_idx, vehicle_id in enumerate(vehicle_ids):
            
            try:
                # clear previous car and spawn new one
                handler.destroy_all_vehicles()
                handler.world_tick(10)
                handler.spawn_vehicle(vehicle_id)
                handler.update_view('3d')
                time.sleep(1)
            except Exception as e:
                print(f"Skipping {vehicle_id} due to spawn error: {e}")
                skipped = locations_per_car * len(distances) * len(pitches) * shots_per_orbit
                pbar.update(skipped)
                continue

            # iterate over spawn points / locations
            chosen_spawns = random.sample(range(n_spawn_points), min(locations_per_car, n_spawn_points))
            
            for spawn_point in chosen_spawns:
                handler.change_spawn_point(spawn_point)
                randomize_lighting(handler)
                # tick to let car settle
                handler.world_tick(50) 
                
                # loops for orbiting areound car
                for dist in distances:
                    for pitch in pitches:
                        orbit_yaws = range(0, 360, yaw_step)
                        
                        for yaw in orbit_yaws:
                            try:
                                set_camera(handler, dist, pitch, yaw)
                                
                                meta = [dist, pitch, yaw, vehicle_id]
                                success = capture_sample(handler, global_counter, meta)
                                
                                if success:
                                    global_counter += 1


                                pbar.update(1)
                                    
                            except Exception as e:
                                handler.world_tick(10)
                                pbar.update(1)

    print(f"\n Data collection complete. Total samples: {global_counter}")
    handler.destroy_all_vehicles()


if __name__ == '__main__':
    main()