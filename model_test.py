from ultralytics import YOLO
import cv2
#Using this forthe experiments section, just to run different models for screenshots
def main():

    model = YOLO(r"runs/segment/train/weights/best.pt")


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    #Set res
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break


        results = model(frame, verbose=False)
        r = results[0]


        annotated_frame = r.plot()   #draws boxes/masks/labels

        #Show the annotated frame
        cv2.imshow("YOLO Webcam", annotated_frame)

        #Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
