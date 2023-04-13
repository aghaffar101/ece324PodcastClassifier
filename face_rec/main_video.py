import simple_facerec

# Encode faces from a folder
sfr = simple_facerec.SimpleFacerec()
sfr.load_encoding_images("face_rec\images")

def findImage(img_path):
    faces_recognized = []

    face_locations, face_names = sfr.detect_known_faces(img_path)
    for face_loc, name in zip(face_locations, face_names):
        print(name)

        if (name != "Unknown" and name not in faces_recognized):
            faces_recognized.append(name)
    
findImage(r"face_rec\testadin.png")