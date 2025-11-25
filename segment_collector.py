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
diff_colour = (124, 14, 14)  # BGR format
sampleSize = 3


def setEnvironment(transforms):
    handler, spawnNo, distance, pitch, yaw, sun_altitude, sun_azimuth = transforms

    handler.change_spawn_point(spawnNo)
    handler.update_distance(distance)
    handler.update_pitch(pitch)
    handler.update_yaw(yaw)
    handler.update_sun_altitude_angle(sun_altitude)
    handler.update_sun_azimuth_angle(sun_azimuth)
    handler.world_tick(100)
    time.sleep(0.1)


def main():
    
    # Initialize CARLA
    try:
        handler, n_spawn_points = initialize_carla()
    except Exception as e:
        print(f"Failed to initialize CARLA: {e}")
        print("Exiting...")
        return
    
    sample_n = 0

    while sample_n < sampleSize:
        try:
            #generate a random camera/environment setup
            spawnPoint = random.randint(1, n_spawn_points-1)
            distance = random.randint(5, 10)
            pitch = random.randint(0, 60)
            yaw = random.randint(0, 359)
            sun_altitude = random.randint(20, 150)
            sun_azimuth = random.randint(0, 360)

            transforms = (handler, spawnPoint, distance, pitch, yaw, sun_altitude, sun_azimuth)
            setEnvironment(transforms)

            # set car colour & get image
            handler.change_vehicle_color(car_colour)
            handler.world_tick(100)
            time.sleep(0.1)
            ref_image = handler.get_image()  # BGR format


            # Generate vehicle mask
            seg_image = handler.get_segmentation()  # BGR format
            # Create vehicle mask (blue pixels in segmentation)
            vehicle_mask = (seg_image[:,:,0] == 255) & \
                        (seg_image[:,:,1] == 0) & \
                        (seg_image[:,:,2] == 0)
            # Validate vehicle mask
            if not validate_car_mask(vehicle_mask):
                print(f"⚠️ Sample {sample_n}: Invalid mask (multiple cars or no car detected)")
                continue


            # change car colour & get image
            handler.change_vehicle_color(diff_colour)
            handler.world_tick(100)
            time.sleep(0.1)
            # Get the actual rendered BGR image
            cross_ref_image = handler.get_image()  # BGR format

            # get intersection of 2 car images and extract car pixels
            intersection_mask = np.isclose(ref_image,cross_ref_image, atol=1e-2).all(axis=-1) # all pixels the same in both images (windows, env)

            #apply vehicle mask to intersection mask to get just windows
            feature_mask = np.zeros_like(intersection_mask)
            feature_mask[vehicle_mask] = intersection_mask[vehicle_mask]

            # Overlay: only keep intersecting pixels in prediction
            feature_overlay = np.where(feature_mask, ref_image, 0)



            # store ref_image, transforms, vehicle_mask, feature_overlay
            # Create a directory for this sample
            out_dir = f"./output_samples/sample_{sample_n}"
            os.makedirs(out_dir, exist_ok=True)

            # Save reference image
            ref_image_path = os.path.join(out_dir, "ref_image.png")
            cv2.imwrite(ref_image_path, ref_image)

            # Save transforms (camera/environment settings)
            transforms_path = os.path.join(out_dir, "transforms.npy")
            # Store as numpy array: [distance, pitch, yaw]
            np.save(transforms_path, np.array([distance, pitch, yaw]))

            # Save vehicle mask
            vehicle_mask_path = os.path.join(out_dir, "vehicle_mask.png")
            # Convert mask to uint8 for view/save (multiply by 255)
            vehicle_mask_uint8 = (vehicle_mask.astype(np.uint8)) * 255
            cv2.imwrite(vehicle_mask_path, vehicle_mask_uint8)

            # Save feature overlay
            feature_overlay_path = os.path.join(out_dir, "feature_overlay.png")
            cv2.imwrite(feature_overlay_path, feature_overlay)

            print(f"✅ Sample {sample_n}")
            sample_n += 1

        except Exception as e:
            print(f"Error generating sample {sample_n}: {e}")
            continue


def validate_car_mask(mask):
    # Convert to uint8
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # 1 label = no car
    # 2 labels = 1 car
    # >2 labels = multiple cars/objects
    return num_labels == 2


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