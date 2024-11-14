import cv2

video=cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

id=input("enter your name: ")
count=0
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,h,w) in faces:
        count=count+1
        cv2.imwrite('datasets/Users.'+str(id)+"."+str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)

    cv2.imshow("Frame",frame)

    k=cv2.waitKey(1)

    if count>100:
        break
video.release()
cv2.destroyAllWindows() 
print("DataSet Collected!.....")   