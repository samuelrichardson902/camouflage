import numpy as np
import cv2
import time
import random
import os
import sys

# Try/Except to handle if you run this without the handler file present
try:
    from CarlaHandler import *
except ImportError:
    print("Error: CarlaHandler.py not found. Please ensure it is in the same directory.")
    sys.exit(1)


# Configuration
output_root = "./dataset_structured"

vehicle_ids = [
    'vehicle.tesla.model3', 
    'vehicle.audi.tt', 
    'vehicle.toyota.prius', 
    'vehicle.jeep.wrangler_rubicon', 
    'vehicle.nissan.patrol'
]

distances= [5,7,10] # camera distances
pitches= [10,30,45] # camera height (degrees)
yaw_step = 15 # degrees between shots 360/15=24 shots
locations_per_car = 25 # how many different spawns to test each car


town_id = 'Town03'
res = 500
car_colour = (124, 124, 124)  # grey reference
diff_colour = (124, 14, 14)   # for differential mask


def ensure_folders():
    """Creates the folder structure if it doesn't exist."""
    subfolders = ['reference', 'masks', 'overlays', 'transforms']
    for folder in subfolders:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)


def set_camera_and_lighting(handler, distance, pitch, yaw):
    """Updates camera pose and randomizes lighting."""
    handler.update_distance(distance)
    handler.update_pitch(pitch)
    handler.update_yaw(yaw)
    
    # Randomize lighting for every single shot to prevent overfitting to specific shadows
    sun_alt = random.randint(15, 90)   # Avoid 0-15 (too dark/night)
    sun_azi = random.randint(0, 360)
    
    handler.update_sun_altitude_angle(sun_alt)
    handler.update_sun_azimuth_angle(sun_azi)
    
    # Force clear weather for clean masks
    handler.update_cloudiness(0.0)
    handler.update_wind_intensity(0.0)
    handler.update_precipitation(0.0)
    handler.world_tick(5) # Tick to apply light changes


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

    # --- OUTER LOOP: VEHICLES ---
    for v_idx, vehicle_id in enumerate(vehicle_ids):
        print(f"\n🚗 Starting collection for vehicle: {vehicle_id} ({v_idx+1}/{len(vehicle_ids)})")
        
        try:
            # Clean up previous car and spawn new one
            handler.destroy_all_vehicles()
            handler.world_tick(10)
            handler.spawn_vehicle(vehicle_id)
            handler.update_view('3d')
            time.sleep(1)
        except Exception as e:
            print(f"Skipping {vehicle_id} due to spawn error: {e}")
            continue

        # --- LOOP: SPAWN LOCATIONS ---
        # Pick 'locations_per_car' random spots to place the car
        chosen_spawns = random.sample(range(n_spawn_points), min(locations_per_car, n_spawn_points))
        
        for spawn_point in chosen_spawns:
            handler.change_spawn_point(spawn_point)
            # Big tick to let car settle on the ground
            handler.world_tick(50) 
            
            # --- LOOP: ORBITING (Distance -> Pitch -> Yaw) ---
            for dist in distances:
                for pitch in pitches:
                    # Generate Orbit Yaws (0, 30, 60 ... 330)
                    orbit_yaws = range(0, 360, yaw_step)
                    
                    for yaw in orbit_yaws:
                        try:
                            # 1. Set Scene
                            set_camera_and_lighting(handler, dist, pitch, yaw)
                            
                            # 2. Capture
                            meta = [dist, pitch, yaw, vehicle_id]
                            success = capture_sample(handler, global_counter, meta)
                            
                            if success:
                                print(f"Sample {global_counter} | {vehicle_id} | D:{dist} P:{pitch} Y:{yaw}")
                                global_counter += 1
                            else:
                                print(f"Sample {global_counter} Failed (Mask Issue)")
                                
                        except Exception as e:
                            print(f"Error on sample {global_counter}: {e}")
                            # Try to recover by ticking
                            handler.world_tick(10)

    print(f"\n✅ Data collection complete. Total samples: {global_counter}")
    handler.destroy_all_vehicles()


if __name__ == '__main__':
    main()