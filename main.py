import cv2

# Integer for the webcam to be used or the string URL of the IP camera
VIDEO_STREAM = 0

def showCapture():
    # Initiate the video capture
    cam = cv2.VideoCapture(VIDEO_STREAM)
    # Initiate frame1 (the first frame for comparison in each iteration) as None (because nothing has been captured yet)
    frame1 = None

    # The streaming loop
    while cam.isOpened():
        # Second frame for comparison
        _, f2 = cam.read()
        # Operate only if the first frame exists (there are at least two frames for comparison)
        if not (frame1 is None):
            # Get the copy of the second frame (f2) so that these operations don't affect the original second frame
            frame2 = f2.copy()

            # Get the grayscale absolute difference of both the first and the second frames to obtain an image corresponding to the motion that took place between both the frames
            diff = cv2.cvtColor(cv2.absdiff(frame1, frame2), cv2.COLOR_RGB2GRAY)

            # Blur the difference image and apply a threshold to sharpen the objects in motion
            threshold, thresholdImage = cv2.threshold(cv2.GaussianBlur(diff, (5, 5), 0), 20, 255, cv2.THRESH_BINARY)

            # Execute morphological operation (dilation) to densify the boundary pixels of the objects in motion
            dilated = cv2.dilate(thresholdImage, None, iterations = 10)

            # Get the contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw only those contour boundaries whose area is larger (rectangles around only those objects which are in significant mortion; small motion is neglected)
            for c in contours:
                if cv2.contourArea(c) < 20000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Quit if the key "q" is pressed
            if cv2.waitKey(10) & 0xff == ord("q"):
                break

            # Show the stream along with the drawn rectangles
            cv2.imshow("Motion detection", frame2)

        # Assign the second frame to frame1 for the next iteration
        frame1 = f2

showCapture()