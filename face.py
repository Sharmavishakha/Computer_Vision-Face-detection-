import cv2

cap = cv2.VideoCapture(0)
cascade_classifier=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
while True:

	ret, frame = cap.read()
	frame=cv2.cvtColor(frame,0)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detections=cascade_classifier.detectMultiScale(gray, 1.3, 10)

	if(len(detections)>0):
		(x,y,w,h) = detections[0]
		frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,50,210),2)       #frame is image captured,(x,y)->coordinates of top left corner and next is for bottom right then color of box(B,G,R) and then thickness of box

	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()