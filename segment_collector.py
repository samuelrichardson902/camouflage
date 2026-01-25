import numpy as np
import cv2
import time
import random
from CarlaHandler import *
import os

# Configurable Parameters
vehicle_id = 'vehicle.tesla.model3'
town_id = 'Town03'
res = 500
car_colour = (124, 124, 124)  # BGR format
diff_colour = (124, 14, 14)   # BGR format
sampleSize = 10
output_root = "./sample_dataset"


def setEnvironment(transforms):
    handler, spawnNo, distance, pitch, yaw, sun_altitude, sun_azimuth = transforms

    handler.change_spawn_point(spawnNo)
    handler.update_distance(distance)
    handler.update_pitch(pitch)
    handler.update_yaw(yaw)
    
    # Lighting
    handler.update_sun_altitude_angle(sun_altitude)
    handler.update_sun_azimuth_angle(sun_azimuth)
    
    # FORCE STATIC WEATHER (Crucial for differential rendering)
    # If clouds/wind move between the two photos, the background will 'change' 
    # and ruin the mask.
    handler.update_cloudiness(0.0)
    handler.update_wind_intensity(0.0)
    handler.update_precipitation(0.0)
    handler.update_fog_density(0.0)
    
    # Tick to apply weather
    handler.world_tick(10)


def process_car_mask(mask, min_pixels=700):
    """
    Cleans the mask by keeping only the largest object (the main car).
    Removes tiny blobs/noise that cause 'multiple vehicles' errors.
    Returns: (is_valid, cleaned_mask)
    """
    # Ensure mask is uint8 (0-255)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # num_labels includes background (0). So we need at least 2 labels (bg + 1 object)
    if num_labels < 2:
        return False, mask 

    # Sort components by Area (column 4) in descending order.
    # We skip index 0 because that is the background.
    # argsort returns indices of the sorted array, so we adjust +1 to match original labels.
    component_indices = np.argsort(stats[1:, 4])[::-1] + 1
    
    # The largest blob (candidate for the car)
    largest_label = component_indices[0]
    largest_area = stats[largest_label, 4]

    # Safety check: Is the largest thing actually a car-sized object?
    if largest_area < min_pixels:
        return False, mask # Largest thing is just a speck of noise
    
    # Create a new mask keeping ONLY the largest component
    cleaned_mask = (labels == largest_label).astype(np.uint8) * 255
    
    return True, cleaned_mask


def main():
    
    # Initialize CARLA
    try:
        handler, n_spawn_points = initialize_carla()
    except Exception as e:
        print(f"Failed to initialize CARLA: {e}")
        print("Exiting...")
        return


    os.makedirs(os.path.join(output_root, 'reference'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'transforms'), exist_ok=True)
    
    sample_n = 0

    while sample_n < sampleSize:
        try:
            # 1. Randomize Environment
            spawnPoint = random.randint(1, n_spawn_points-1)
            distance = random.randint(5, 10)
            pitch = random.randint(0, 60)
            yaw = random.randint(0, 359)
            sun_altitude = random.randint(20, 150)
            sun_azimuth = random.randint(0, 360)

            transforms = (handler, spawnPoint, distance, pitch, yaw, sun_altitude, sun_azimuth)
            setEnvironment(transforms)

            # --- STEP 2: REFERENCE IMAGE (GREY) ---
            handler.change_vehicle_color(car_colour)
            
            # CRITICAL: Wait for car to fall and settle.
            # Since change_vehicle_color respawns the car, it falls from slight height.
            # If we don't wait enough, the car will be in different positions in the two photos.
            handler.world_tick(50) 
            time.sleep(0.05)
            
            ref_image = handler.get_image()  # BGR format (500, 500, 3)


            # --- STEP 3: SEGMENTATION MASK ---
            seg_image = handler.get_segmentation()
            
            # Create raw mask (blue pixels)
            raw_mask = (seg_image[:,:,0] == 255) & \
                       (seg_image[:,:,1] == 0) & \
                       (seg_image[:,:,2] == 0)
            
            # Clean and Validate mask (Fixes "Multiple Vehicles" and "Tiny Blob" errors)
            is_valid, vehicle_mask = process_car_mask(raw_mask)

            if not is_valid:
                print(f"⚠️ Sample {sample_n}: Invalid mask (no distinct car detected)")
                continue


            # --- STEP 4: DIFFERENTIAL IMAGE (RED) ---
            handler.change_vehicle_color(diff_colour)
            
            # Wait exactly the same amount for physics to settle
            handler.world_tick(50)
            time.sleep(0.05)
            
            cross_ref_image = handler.get_image()  # BGR format

            # --- STEP 5: FEATURE EXTRACTION ---
            # Get intersection. 
            # Increased atol=4 ensures minor JPEG/Render noise doesn't break the mask.
            intersection_mask = np.isclose(ref_image, cross_ref_image, atol=4).all(axis=-1)

            # Apply vehicle mask to intersection to isolate windows/trim
            # (We only care about intersection pixels INSIDE the car silhouette)
            feature_mask = np.zeros_like(intersection_mask)
            
            # Convert mask to boolean for indexing
            vehicle_mask_bool = vehicle_mask > 0
            feature_mask[vehicle_mask_bool] = intersection_mask[vehicle_mask_bool]

            # --- FIX: BROADCASTING ERROR ---
            # feature_mask is (500, 500). ref_image is (500, 500, 3).
            # We add a dimension to make mask (500, 500, 1) so it broadcasts correctly.
            feature_overlay = np.where(feature_mask[:, :, np.newaxis], ref_image, 0)


            # --- CHANGED: SAVE DATA TO TYPE FOLDERS ---
            # Save Reference Image
            cv2.imwrite(f"{output_root}/reference/{sample_n}.png", ref_image)

            # Save Transforms (Metadata)
            np.save(f"{output_root}/transforms/{sample_n}.npy", np.array([distance, pitch, yaw]))

            # Save Mask
            cv2.imwrite(f"{output_root}/masks/{sample_n}.png", vehicle_mask)

            # Save Feature Overlay
            cv2.imwrite(f"{output_root}/overlays/{sample_n}.png", feature_overlay)

            print(f"✅ Sample {sample_n}")
            sample_n += 1

        except Exception as e:
            print(f"Error generating sample {sample_n}: {e}")
            # Optional: Print traceback to see line number of errors
            # import traceback
            # traceback.print_exc()
            continue


def initialize_carla():
        """
        Initialize CARLA and spawn a vehicle.
        """
        try:
            handler = CarlaHandler(x_res=res, y_res=res, town=town_id)
            time.sleep(2)  # Allow time for initialization
            handler.world_tick(10)
            handler.destroy_all_vehicles()
            handler.world_tick(100)

            # Spawn vehicle
            handler.spawn_vehicle(vehicle_id)
            handler.update_view('3d')
            n_spawn_points = handler.get_spawn_points()
            return handler, n_spawn_points
        except Exception as e:
            print(f"Error initializing CARLA: {e}")
            print("Make sure CARLA simulator is running and ready on localhost:2000")
            raise


if __name__ == '__main__':
    main()