#! /usr/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import os
import cv2

 
file_identifier = '2020-09-17-18-34-53'

fwd_dir = './calib_images_'+ file_identifier
if not os.path.isdir(fwd_dir):
    os.mkdir(fwd_dir)

# Instantiate CvBridge
bridge = CvBridge()
count = 1
def image_callback(msg):
    global count
    print("Received an image! ",count)
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite(fwd_dir+'/image_{}.jpg'.format(count), cv2_img)
    count+=1


def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()