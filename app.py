#Comment line 322 before demonstrating app.py !!!!!!!

import os,cv2,time
from flask import Flask,render_template,Response
import math
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

app=Flask(__name__,static_url_path='/static')

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Retrieve the height and width of the input image.
    height, width, _ = output_image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks :
        # Draw Pose landmarks on the output image.
        #mp_drawing.draw_landmarks(output_image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
        
        # 1. Draw face landmarks
        '''mp_drawing.draw_landmarks(output_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, None,
                                 #mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )'''
        
        # 2. Right hand
        mp_drawing.draw_landmarks(output_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(output_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),(landmark.z * width)))
    else: 
        cv2.putText(output_image, "Make Sure Full body visible", (568,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            
# Return the output image and the found landmarks.
    return output_image, landmarks

# Calculating Angles
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

# Classifying Poses
def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value])
    
    #Get coordinates.
    left_elbow = list(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value])
    right_elbow = list(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value])
    left_shoulder = list(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value])
    right_shoulder = list(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value])
    left_hip = list(landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value])
    right_hip = list(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value])
    left_knee = list(landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value])
    right_knee = list(landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value])
    
    # Visualize angle
    cv2.putText(output_image, str(int(left_elbow_angle)), (left_elbow[0],  left_elbow[1]),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_image, str(int(right_elbow_angle)),(right_elbow[0], right_elbow[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_image, str(int(left_knee_angle)),(left_knee[0],   left_knee[1]),     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_image, str(int(right_knee_angle)),(right_knee[0], right_knee[1]),    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
        
    # Return the output image and the classified label.
    return output_image, label
# Pose Estimation
def gen_frames():
    # Pose Estimation

    # Setup Holistic Pose function for video.
    pose_video = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

    # Setup Pose function for video.
    #pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

    # Initialize a resizable window.
    #cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    fps = []

    # Initialize a variable to store the time of the previous frame.
    time1 = 0

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():
        
        # Read a frame.
        ok, frame = camera_video.read()
        
        # Check if frame is not read properly.
        if not ok:
            
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame,pose_video ,display=False)
        
        # Check if the landmarks are detected.
        if landmarks:
            
            # Perform the Pose Classification.
            frame, _ = classifyPose(landmarks, frame, display=False)
        
        # Set the time for this frame to the current time.
        time2 = time.time()
        
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:
        
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)
            
            fps.append(int(frames_per_second))
            
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (0, 100),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def signupPage():
    return(render_template('signup.html'))

@app.route('/Profile.html')
def profile():
    return(render_template('Profile.html'))

@app.route('/About.html')
def about():
    return(render_template('About.html'))

@app.route('/Home.html')
def home():
    return(render_template('Home.html'))

@app.route('/Training.html')
def training():
    return(render_template('Training.html'))

@app.route('/TPose.html')
def tpose():
    return(render_template('TPose.html'))    

@app.route('/Tree.html')
def tree():
    return(render_template('Tree.html'))  

@app.route('/Warrior-2.html')
def warrior2():
    return(render_template('Warrior-2.html'))  

@app.route('/Training.html')
def back1():
    return(render_template('Training.html'))   

@app.route('/Tree.html')
def back2():
    return(render_template('Training.html')) 

@app.route('/Warrior2.html')
def back3():
    return(render_template('Training.html'))

@app.route('/T')
def Start1():  
    #os.system('python ShapeWell.py')
    #return(render_template('Tpose.html'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Tree')
def Start2():
    #os.system('python ShapeWell.py')
    #return(render_template('Tree.html'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/W2')
def Start3():
    #os.system('python ShapeWell.py')
    #return(render_template('Warrior2.html'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
if (__name__)=="__main__":
    port = int(os.environ.get("PORT", 5000)) # <-----
    app.run(host='0.0.0.0', port=port)
