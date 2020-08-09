import numpy as np
import cv2

#face,eye,mouth haar model road
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
video_capture = cv2.VideoCapture(0)
t = 0
DFR = 0
initExist = False

# when recognition of the left and right eyes is reversed, processed as a conditional statement.
def eyeInverter(ex,ey,ew,eh,lx,ly,lw,lh):
    if lx<ex:
        #cv2.rectangle(roi_color,(lx,ly),(ex+ew,ly+lh),(0,0,0),2)
        eyelength = ex+ew-lx
    else:
        #cv2.rectangle(roi_color,(ex,ey),(lx+lw,ly+lh),(0,0,0),2)
        eyelength = lx+lw-ex
    return eyelength

while True:
    # reading webcame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Put the face location information in a numpy array called faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors = 5, minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # With a for statement so that multiple faces can be recognized
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        # Right eye and left eye recognition
        reye = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors = 5, minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )

        leye = leye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors = 5, minSize = (27,30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )

        ok = True
        try:
            ex,ey,ew,eh = reye[0]
            lx,ly,lw,lh = leye[0]
        except:
            pass
            ok = False

        eyelength = 0
        # If leye and reye cannot recognize or read the same eye, use the eye arrangement of the previous frame
        if ok and abs(ex-lx) >= 30:
            initExist = True
            preLeye = leye[0]
            preReye = reye[0]
            eyelength = eyeInverter(ex,ey,ew,eh,lx,ly,lw,lh)
        elif initExist == True:
            ex,ey,ew,eh = preReye
            lx,ly,lw,lh = preLeye
            eyelength = eyeInverter(ex,ey,ew,eh,lx,ly,lw,lh)
        elif initExist == False:
            pass

        #reading mouth
        mouth = mouth_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors = 3, minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
        # Exception handling when mouth is not recognized
        premouth = 0
        try:
            mins = np.apply_along_axis(lambda a: np.argmax(a), 0, mouth)
        except:
            if premouth == 0:
                pass
            else:
                mins = np.apply_along_axis(lambda a: np.argmax(a), 0, premouth)
                mx,my,mw,mh = premouth[mins[1]]
        else:
            mx,my,mw,mh = mouth[mins[1]]
            premouth = mouth

        #cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
        mouthlength = mw
    t+=1

    # face angle calculate & putText
    if t<15:
        cv2.putText(frame, "Please keep 0 degrees", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
    elif t== 15:
        DFR = eyelength/mouthlength
    else:
        CFR = eyelength/mouthlength
        faceAngle = (CFR-DFR)/0.05
        faceAngle = "faceAngle : " + str(round(faceAngle,3))
        cv2.putText(frame, faceAngle, (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()