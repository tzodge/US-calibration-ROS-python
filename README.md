# US-calibration-ROS-python
Copy the contents of [rosbag_data](https://drive.google.com/drive/folders/1ItOMB5gcbs07oUv2EPJTvkaGrKmgX-N1?usp=sharing) in the folder rosbag_data 



For calculations, refer to [presentation](https://docs.google.com/presentation/d/1E5cLrZCGC0eLe_q-Whda-CIxITOYS3GRCLCoh0wb2Lg/edit?usp=sharing)
 

This module assumes that the rosbags have unsynched pose_ee and raw image data.


# 1. Extract, time sync and save the data
the rosbag for \transf1 can be found at [rosbags](https://drive.google.com/drive/folders/1CdlzdJTs855xpAYpTy7xYjVWpYWExtpZ?usp=sharing)

Run 
``` python

python process_rosbag_clean.py <rosbag path>  <data save location>

# for example
python process_rosbag_clean.py ./rosbags/calibration_bag_transform1_2020-09-17-18-38-33.bag  ./rosbag_data/transf1


```

# 2. Select images 
Go to /transf1/calib_images and put the ids of those numbers in 
/transf1/extra_files/selected_image_ids.txt. 


# 3. Annotate z wire phantom data
Images will appear 
select points 1 to 9 for each image and press escape
``` python
python annotate_image_data_clean.py ./rosbag_data/transf1/

```

# 4. Estimate camera pose from annotated data
Run 
``` python
python frame_from_coords_clean.py ./rosbag_data/transf1/

```

# 5. Calibrate
Run 
``` python
python calibrate.py ./rosbag_data/transf1/

```

