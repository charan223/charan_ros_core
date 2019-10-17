#!/usr/bin/env python
import rospy
import numpy as np
from duckietown_msgs.msg import Twist2DStamped, LanePose, SegmentList, Segment

def filtered_seglist_cb(seglist_msg):
        
        #initialize variables
        yellow_offset, white_offset, omega_gain = -0.15, 0.15, 4
        white_seg_count, yellow_seg_count = 0, 0
        white_x_accumulator, white_y_accumulator, yellow_x_accumulator, yellow_y_accumulator = 0.0, 0.0, 0.0, 0.0
        white_centroid_x, white_centroid_y, yellow_centroid_x, yellow_centroid_y = 0.0, 0.0, 0.0, 0.0

        for segment in seglist_msg.segments:
            #the point is behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0: 
                continue

            #calculate white segments sum, count values
            if segment.color == segment.WHITE:
                white_x_accumulator += (segment.points[0].x + segment.points[1].x) / 2
                white_y_accumulator += (segment.points[0].y + segment.points[1].y) / 2 
                white_seg_count += 1.0
            #calculate yellow segments sum, count values
            elif segment.color == segment.YELLOW:
                yellow_x_accumulator += (segment.points[0].x + segment.points[1].x) / 2
                yellow_y_accumulator += (segment.points[0].y + segment.points[1].y) / 2 
                yellow_seg_count += 1.0
            #skip red segments
            else:
                continue

        #calculate centroid for white segments
        if white_seg_count > 0:
            white_centroid_x, white_centroid_y = white_x_accumulator/white_seg_count, white_y_accumulator/white_seg_count

        #calculate centroid for yellow segments
        if yellow_seg_count > 0:
            yellow_centroid_x, yellow_centroid_y = yellow_x_accumulator/yellow_seg_count, yellow_y_accumulator/yellow_seg_count

        #if white seg count is greater, trust white line segments
        if  white_seg_count >  yellow_seg_count:   
            follow_point_x = white_centroid_x
            follow_point_y = white_centroid_y + white_offset

        #if yellow seg count is greater, trust yellow line segments
        elif  yellow_seg_count > white_seg_count:  
            follow_point_x = yellow_centroid_x
            follow_point_y = yellow_centroid_y + yellow_offset
        
        #if both are equal, take average
        else:
            follow_point_x = 0.5 * (white_centroid_x + yellow_centroid_x)
            follow_point_y = 0.5 * (white_centroid_y + yellow_centroid_y)
            #check if they are zero, because they might become zero if no white/yellow segments are encountered
            if follow_point_x == 0 and follow_point_y == 0:
                follow_point_x, follow_point_y = 0.1, 0

        #tan_alpha = y/x => alpha = tan-1(y/x)
        alpha = np.arctan2(follow_point_y, follow_point_x)
        lookahead_dist = np.sqrt(follow_point_x * follow_point_x + follow_point_y * follow_point_y)
        #calculating v, omega
        v = 0.25
        omega  =  2 * v * np.sin(alpha) / lookahead_dist

        #publishing to car_cmd topic
        car_control_msg = Twist2DStamped()
        car_control_msg.v = v
        car_control_msg.omega = omega * omega_gain
        pub_car_cmd.publish(car_control_msg)

    
if __name__ == "__main__":
    #defining node, publisher, subscriber
    rospy.init_node("purepursuit_controller_node", anonymous=True)
    sub_filtered_seglist = rospy.Subscriber("/bayesianduckie/lane_filter_node/seglist_filtered", SegmentList, filtered_seglist_cb)
    pub_car_cmd = rospy.Publisher("/bayesianduckie/joy_mapper_node/car_cmd", Twist2DStamped, queue_size=10)

    rospy.spin()

    
    
    
    
    
 