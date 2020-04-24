# smart_home_automation_with_segmented_image_processing
code for the paper smart home automation with segmented image processing
The idea is to save energy by controlling lights by detecting humans<br>
The room is divided into quadrant and each quadrant is linked to a relay which controls electrical appliances in that specific quadrant
<br>The relay is connected to an arduino and a esp8266 for http communication
<br>The camera is connected to a raspberry pi 3b+ which acts as the brain of the system. The camera captures frames and places in the buffer
<br>The python script picks a frame each time and divides it into quadrant. Human detection is run and the co-ordinates of humans are stored. The co-ordinates are processed to find the quadrant in which the human lies. 
<br>The arduino is sent a http signal via esp8266 to change the status of the electrical appliances only in those quadrants
<br>OpenCV is used for image processing and Tensorflow for Human detection using SSD Mobilenet 
