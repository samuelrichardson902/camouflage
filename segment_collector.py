import numpy as np
import cv2
import time
import random
import os
import sys
from tqdm import tqdm

# Try/Except to handle if you run this without the handler file present
try:
    from CarlaHandler import *
except ImportError:
    print("Error: CarlaHandler.py not found. Please ensure it is in the same directory.")
    sys.exit(1)


# Configuration
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
    """Creates the folder structure if it doesn't exist."""
    subfolders = ['reference', 'masks', 'overlays', 'transforms']
    for folder in subfolders:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)


def randomize_lighting(handler):
    """Randomize lighting for every position to prevent overfitting to specific shadows"""
    sun_alt = random.randint(15, 90)
    sun_azi = random.randint(0, 360)
    
    handler.update_sun_altitude_angle(sun_alt)
    handler.update_sun_azimuth_angle(sun_azi)
    
    # Force clear weather for clean masks
    handler.update_cloudiness(0.0)
    handler.update_wind_intensity(0.0)
    handler.update_precipitation(0.0)
    handler.world_tick(5) # Tick to apply light changes


def set_camera(handler, distance, pitch, yaw):
    """Updates camera pose"""
    handler.update_distance(distance)
    handler.update_pitch(pitch)
    handler.update_yaw(yaw)
    handler.world_tick(5)
    time.sleep(0.02)
    
    


def process_car_mask(mask, min_pixels=700):
    """
    Standard cleaning: Keeps only the largest connected component (the car).
    """
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2: return False, mask 

    # Sort by area (descending), ignore index 0 (background)
    component_indices = np.argsort(stats[1:, 4])[::-1] + 1
    largest_label = component_indices[0]
    
    if stats[largest_label, 4] < min_pixels: return False, mask
    
    cleaned_mask = (labels == largest_label).astype(np.uint8) * 255
    return True, cleaned_mask


def capture_sample(handler, save_idx, meta_data):
    """
    Performs the Reference -> Mask -> Color Change -> Diff workflow.
    Returns True if successful.
    """
    # A. REFERENCE IMAGE (Grey)
    handler.change_vehicle_color(car_colour)
    # Wait for physics to settle after respawn/color change
    handler.world_tick(20) 
    time.sleep(0.05)
    
    ref_image = handler.get_image()
    
    # B. SEGMENTATION MASK
    seg_image = handler.get_segmentation()
    # Blue channel usually contains vehicles in CARLA default color conversion
    raw_mask = (seg_image[:,:,0] == 255) & (seg_image[:,:,1] == 0) & (seg_image[:,:,2] == 0)
    is_valid, vehicle_mask = process_car_mask(raw_mask)

    if not is_valid: return False

    # C. DIFFERENTIAL IMAGE (Red)
    handler.change_vehicle_color(diff_colour)
    # Wait EXACTLY the same amount as before to minimize physics drift
    handler.world_tick(20)
    time.sleep(0.05)
    
    cross_ref_image = handler.get_image()

    # D. FEATURE OVERLAY GENERATION
    # 1. Find pixels that changed significantly (The car body) vs those that didn't (Windows/Lights/Bg)
    # We want parts that DID NOT change (windows/lights) but are INSIDE the mask.
    
    # "isclose" finds pixels that are same in both (Windows, BG)
    intersection = np.isclose(ref_image, cross_ref_image, atol=6).all(axis=-1)
    
    # Apply vehicle mask: We only want "Same Pixels" that are "Inside Car"
    feature_mask = np.zeros_like(intersection)
    feature_mask[vehicle_mask > 0] = intersection[vehicle_mask > 0]
    
    # Create visual overlay
    feature_overlay = np.where(feature_mask[:, :, np.newaxis], ref_image, 0)

    # E. SAVE DATA
    filename = f"{save_idx:05d}.png"
    cv2.imwrite(f"{output_root}/reference/{filename}", ref_image)
    cv2.imwrite(f"{output_root}/masks/{filename}", vehicle_mask)
    cv2.imwrite(f"{output_root}/overlays/{filename}", feature_overlay)
    
    # Save Metadata (Distance, Pitch, Yaw, VehicleName)
    # meta_data is [distance, pitch, yaw, vehicle_id_string]
    np.save(f"{output_root}/transforms/{save_idx:05d}.npy", np.array(meta_data))
    
    return True

def main():
    ensure_folders()

    shots_per_orbit = len(range(0,360,yaw_step))
    total_samples = len(vehicle_ids) * locations_per_car * len(distances) * len(pitches) * shots_per_orbit
    print(f"{total_samples} images to be collected")


    # 1. Initialize CARLA once
    try:
        # Note: We init with NO vehicle first, or destroy the default one immediately
        handler = CarlaHandler(x_res=res, y_res=res, town=town_id)
        time.sleep(2)
        handler.world_tick(10)
        n_spawn_points = handler.get_spawn_points()
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        return

    global_counter = 0

    with tqdm(total=total_samples, desc="Collecting Samples", unit="img") as pbar:

        # --- OUTER LOOP: VEHICLES ---
        for v_idx, vehicle_id in enumerate(vehicle_ids):
            
            try:
                # Clean up previous car and spawn new one
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

            # --- LOOP: SPAWN LOCATIONS ---
            # Pick 'locations_per_car' random spots to place the car
            chosen_spawns = random.sample(range(n_spawn_points), min(locations_per_car, n_spawn_points))
            
            for spawn_point in chosen_spawns:
                handler.change_spawn_point(spawn_point)
                randomize_lighting(handler)
                # Big tick to let car settle on the ground
                handler.world_tick(50) 
                
                # orbiting loop (Distance Pitch Yaw)
                for dist in distances:
                    for pitch in pitches:
                        orbit_yaws = range(0, 360, yaw_step)
                        
                        for yaw in orbit_yaws:
                            try:
                                # 1. Set Scene
                                set_camera(handler, dist, pitch, yaw)
                                
                                # 2. Capture
                                meta = [dist, pitch, yaw, vehicle_id]
                                success = capture_sample(handler, global_counter, meta)
                                
                                if success:
                                    global_counter += 1


                                pbar.update(1)
                                    
                            except Exception as e:
                                # Try to recover by ticking
                                handler.world_tick(10)
                                pbar.update(1)

    print(f"\n✅ Data collection complete. Total samples: {global_counter}")
    handler.destroy_all_vehicles()


if __name__ == '__main__':
    main()