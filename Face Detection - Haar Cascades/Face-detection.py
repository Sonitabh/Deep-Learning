import cv2

#%% Parameters

fr = 1          # frame counter
sc = 1          # Picture Scaling

sf = 1.2        # Scaling for detection

#%% Video Detection

cap = cv2.VideoCapture(0)                           # Camera detection
cascPath = "haarcascade_frontalface_default.xml"

# face Cascades
casc = cv2.CascadeClassifier(cascPath)

# Loop for always running camera until 'q' is pressed
while True:
    fr += 1

    # read frame
    check, frame = cap.read()
    fr1 = frame
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # FAce detection
    frame = cv2.resize(frame,(int(frame.shape[1]*sc),int(frame.shape[0]*sc)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = casc.detectMultiScale(gray, scaleFactor=sf, minNeighbors=5)
    
    # Putting markers on the frame
    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),0)
        frame = cv2.circle(frame,(int(x+(w/2)),int(y+(h/2))), 5, (0,0,255), -1)
        frame = cv2.circle(frame,(int(x+(w/2)),int(frame.shape[0]/2)), 5, (255,0,0), -1)
        frame = cv2.circle(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)), 5, (0,255,0), -1)
    
    # Showing the captured frame
    fr1 = frame
    cv2.imshow("cap",frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

