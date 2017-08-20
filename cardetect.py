import cv2

car_cascade = cv2.CascadeClassifier('cars.xml')
# if you want to save the output video uncomment the next two lines and line 16 and line 19
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
cap = cv2.VideoCapture('input.avi')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    car = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in car:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # out.write(gray)
    cv2.imshow('output', img)

    if cv2.waitKey(30) == 27:
        break
cap.release()
# out.release()
cv2.destroyAllWindows()
