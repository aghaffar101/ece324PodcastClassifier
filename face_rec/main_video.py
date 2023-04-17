import cv2
from face_rec import simple_facerec

# Encode faces from a folder
sfr = simple_facerec.SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\james\OneDrive\Documents\GitHub\odin-flask-app\face_rec\images")

def findVideo(vid_path):
    faces_recognized = []

    # Load Video
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()

    count = 0

    print("Start")
    while ret:
        ret, frame = cap.read()
        if ret:
            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                print(name)

                if (name != "Unknown" and name not in faces_recognized):
                    faces_recognized.append(name)
            count += 30 # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            print("Done")

            return faces_recognized
            cap.release()
            break
    

# if __name__ == "__main__":   
#     findVideo(r"C:\Users\james\OneDrive\Documents\GitHub\odin-flask-app\face_rec\2022-09-28_03-41-13_UTC.mp4")