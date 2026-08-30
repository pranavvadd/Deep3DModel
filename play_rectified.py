import cv2
import numpy as np
import os

def play():
    path_L = 'training_pipeline/rectified/rect_L.avi'
    path_R = 'training_pipeline/rectified/rect_R.avi'
    
    capL = cv2.VideoCapture(path_L)
    capR = cv2.VideoCapture(path_R)

    if not capL.isOpened():
        print(f"Error: Could not open {path_L}")
        return

    print("Playing. Press 'q' to quit.")
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        
        if not retL:
            break
        
        # If Right side is broken/empty, create a black frame so we can still compare
        if not retR:
            frameR = np.zeros_like(frameL)

        # Stack Side-by-Side
        combined = np.hstack((frameL, frameR))
        
        # Draw horizontal alignment lines every 50 pixels
        for y in range(0, combined.shape[0], 50):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

        cv2.imshow('Verification: Left vs Right (Look for Green Line Alignment)', combined)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play()
