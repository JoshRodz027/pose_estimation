import cv2
import mediapipe as mp
import numpy as np

from utils import calculate_angle


class ModelPose:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.counter = 0
        self.stage= None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def run(self,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        with self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence) as pose:
            while self.cap.isOpened():
                ret,frame = self.cap.read()
                
                # Recolor image to RGB. Original feed in openCV is BGR! Therefore we need to re-order and put it into mediapipe
                image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image.flags.writeable= False ## -> memory saving

                # Making detection of the image as inference
                results = pose.process(image)

                # Recolouring back image to BGR for openCV
                image.flags.writeable=True
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) ## --> covert back from mediapipe RGB to openCV BGR so that we can ask openCV to display the results

                # Extract Landmarks
                try:
                    landmarks= results.pose_landmarks.landmark
                    # print(landmarks)

                    # extract points/coordinates to lack at to visualise angle
                    shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    #Calculate angle
                    angle = calculate_angle(shoulder,elbow,wrist)

                    #Visualize angle , determining position using array multiplication method using elbow by webcam feed to get exact webcam coordinates that are normalised
                    cv2.putText(image,str(angle),
                                tuple(np.multiply(elbow, [640,480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                                )
                                

                    ## Curl counter logic
                    if angle>110:
                        self.stage ="down"
                    if angle < 55 and self.stage =="down":
                        self.stage = "up"
                        self.counter+=1
                        print(self.counter)
                    
                    
                    
                except Exception as e:
                    ## Sometimes camera feed have no landmark, so to prevent crashing, we let the exception pass
                    # print(e)
                    pass

                ## Visualise curl counter
                ## setup status box
                cv2.rectangle(image,(0,0),(200,73),(245,117,16), -1) # iamge we want to apply to, start point, end point, colour and line width of the status box

                ## Rep data into status box - Start coordinate(15,12), FONT and size, colour and line width and type of text
                cv2.putText(image,"REPS", (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image,str(self.counter), (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                ## Stage data
                cv2.putText(image,"Stage", (65,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image,self.stage, (60,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


                    
                    # Rendering detections on screen . results.pose_landmarks has points of where the landmarks are. mp_pose.POSE_CONNECTIONS shows the pose points that connections are.
                    # whats this doing? Its first taking our image in np.array format, we then pass in our landmarks list(in this case the results). first mp_drawing spec is the circles and 2nd is the connection spec, the lines!
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                            self.mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

                cv2.imshow("mediapipe feed", image)

                ## closes if we close the screen or hit q button
                if cv2.waitKey(10) & 0xFF==ord("q"):
                    break
 
        # destroys video feed
        self.cap.release()
        cv2.destroyAllWindows()
        
    # @staticmethod
    # def caluclate_angle(a,b,c):
    #     a=np.array(a)
    #     b=np.array(b)
    #     c=np.array(c)

    #     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0]) # Y end - y mid and x value then first and mid point.
    #     angle = np.abs(radians*180.0/np.pi)

    #     if angle >180.0:
    #         angle = 360 -angle # adjust angles as our joints cant rotate more than 180

    #     return angle

if __name__=="__main__":
    model=ModelPose() 
    model.run()