import cv2 

video=cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
id=input("Enter your id: ")
#id=int(id)

count=0

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        count=count+1
       
        name="ImageData/"+str(id)+"_"+str(count)+".jpg"
        cv2.imwrite(name,gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    
    cv2.imshow("Capturing",frame)
    key=cv2.waitKey(1)

    if count>500:
        break


video.release()
cv2.destroyAllWindows()
print("Collecting Images Complete")
