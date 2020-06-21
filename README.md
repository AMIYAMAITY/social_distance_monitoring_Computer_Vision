# Social Distance Monitoring for Covid-19
### Overview

I made one ultra simple project or you can say this is my experiment. Basically this project we can use in any shopping mall or any crowd area for monitoring whether people are maintaining social distance or not for covid-19 . If people are not maintaining social distance we can give them alert to keep social distance,also here we can easily identify which area people are very close to each other.

### Technical Overview

Here i have used darknet yolov3 pretrained weight and OpenCv. Yolo used for object detection in this case person detection and OpenCv used for drawing bounding boxes and calculating coordinate systems. This project basically returns three images one original drawn image frame, second bird-eye-view frame, third concat both image frames to make one image frame. I have attached all threes snapshot given below-

#### First- ( Original drawn image frame ):
![alt text](https://github.com/AMIYAMAITY/social_distance_monitoring_Computer_Vision/blob/master/images/social_distance_monitoring_original_frame.gif "Original drawn image frame")

#### Second- ( Bird-eye-view image frame ):
![alt text](https://github.com/AMIYAMAITY/social_distance_monitoring_Computer_Vision/blob/master/images/social_distance_monitoring_bird_eye_view.gif "Bird-eye-view image frame")

#### Third- ( Concat of both images ):
![alt text](https://github.com/AMIYAMAITY/social_distance_monitoring_Computer_Vision/blob/master/images/social_distancing_monitoring_concat.gif "Concat of both images")

#### How to run

```
Pull this repo.
```
```
cd social_distance_monitoring_Computer_Vision
```
##### Before executing below command make sure you need to set ROI in monitoring_from_video function.
```
python3 social_distance_monitoring.py -i=/video_path/People-6387.mp4  -c=/yolov3_config_path/yolov3.cfg -w=/yolov3_weight_path/yolov3.weights -cl=/coco_dataset_names_path/coco.names
```
#### Useful links
###### yolov3.cfg : https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
###### coco.name : https://github.com/pjreddie/darknet/blob/master/data/coco.names
###### yolov3.weight : https://pjreddie.com/media/files/yolov3.weights
###### video: https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/

### What's Next

Here i am calculating close distance by pixel value. You can set whatever pixel you want in line 18 variable name PIXEL_THRESH . But it's not the correct solution in the real scenario , we need to calculate it as real unit like cm,inch,feet like that.

##### Thank you :)
