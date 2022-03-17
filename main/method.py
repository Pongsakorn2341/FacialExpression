
import cv2
class Method:
    roi_gray = []
    def checkSizeImage(self, height, width):
        if(height != 350 or width != 350):
            return False
        else:
            return True
    
    def detectFace(self, gray):
        face_cascade = cv2.CascadeClassifier('./utils/xml/haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier('./utils/xml/haarcascade_eye_tree_eyeglasses.xml')
        smile_cascade = cv2.CascadeClassifier('./utils/xml/haarcascade_smile.xml')

        face = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 4,
            minSize = (200, 200),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in face:
            self.roi_gray = gray[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(
            self.roi_gray,
            scaleFactor = 1.16,
            minNeighbors = 35,
            minSize = (25, 25),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        eyes = eye_cascade.detectMultiScale(self.roi_gray)
        if len(face) == 0 or len(smile) < 1 or len(eyes) < 2:
            return False
        else:
            return True
    
    def checkBlur(self, fm, threshold):
        return fm < threshold

    def getGrayScale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def getVarianceLaplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()
