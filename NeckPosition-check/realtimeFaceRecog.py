import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
initExist = False
video_capture = cv2.VideoCapture(0)
t = 0
DFR = 0

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors = 5, minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
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
        if ok and abs(ex-lx) >= 30:
            initExist = True
            preLeye = leye[0]
            preReye = reye[0]
            if lx<ex:
                cv2.rectangle(roi_color,(lx,ly),(ex+ew,ly+lh),(0,0,0),2)
                eyelength = ex+ew-lx
            else:
                cv2.rectangle(roi_color,(ex,ey),(lx+lw,ly+lh),(0,0,0),2)
                eyelength = lx+lw-ex
        elif initExist == True:
            ex,ey,ew,eh = preReye
            lx,ly,lw,lh = preLeye
            if lx<ex:
                cv2.rectangle(roi_color,(lx,ly),(ex+ew,ly+lh),(0,0,0),2)
                eyelength = ex+ew-lx
            else:
                cv2.rectangle(roi_color,(ex,ey),(lx+lw,ly+lh),(0,0,0),2)
                eyelength = lx+lw-ex
        elif initExist == False:
            pass

        mouth = mouth_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors = 3, minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )
        mins = np.apply_along_axis(lambda a: np.argmax(a), 0, mouth)
        mx,my,mw,mh = mouth[mins[1]]

        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
        mouthlength = mw
        t+=1
        if t<15:
            cv2.putText(roi_color, "Please keep 0 degrees", (0,100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
        elif t== 15:
            DFR = eyelength/mouthlength
        CFR = eyelength/mouthlength
        faceAngle = (CFR-DFR)/0.05
        try:
            cv2.putText(roi_color, faceAngle, (0,100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
        except:
            pass
        print(faceAngle)

    cv2.imshow('Video', frame)



    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()