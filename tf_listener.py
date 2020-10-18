#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import geometry_msgs.msg
import turtlesim.srv

from geometry_msgs.msg import TransformStamped
if __name__ == '__main__':
    rospy.init_node('tf2_turtle_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # rospy.wait_for_service('spawn')
    # spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    # turtle_name = rospy.get_param('turtle', 'turtle2')
    # spawner(4, 2, 0, turtle_name)

    pose_pub = rospy.Publisher('/pose_ee', TransformStamped, queue_size=10)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('base', 'tool0_controller', rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        # msg = geometry_msgs.msg.Twist()

#         msg.angular.z = 4 * math.atan2(trans.transform.translation.y, trans.transform.translation.x)
#         msg.linear.x = 0.5 * math.sqrt(trans.transform.translation.x ** 2 + trans.transform.translation.y ** 2)
# base
        pose_pub.publish(trans)

        rate.sleep()
 