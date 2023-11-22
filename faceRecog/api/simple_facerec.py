import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        face_names = []
        face_accuracies = []
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small_frame)
        # print(face_loc)
        for face_loc in face_locs:
            face_loc = np.array(face_loc)  # Convert the tuple to a NumPy array
            y1, x2, y2, x1 = face_loc
            face_width = x2 - x1
            face_height = y2 - y1
            # print(face_loc[2])
            min_face_size_threshold = (5, 5)

            if (
                face_width >= min_face_size_threshold[0]
                and face_height >= min_face_size_threshold[1]
            ):
                print("enter if")
                y1 = int(y1 / self.frame_resizing)
                x2 = int(x2 / self.frame_resizing)
                y2 = int(y2 / self.frame_resizing)
                x1 = int(x1 / self.frame_resizing)

                # Crop the face from the original frame
                cropped_face = frame[y1:y2, x1:x2]

                face_locations = (
                    0,
                    int(face_width / self.frame_resizing),
                    int(face_height / self.frame_resizing),
                    0,
                )
                face_encodings = face_recognition.face_encodings(
                    cropped_face, [face_locations]
                )

                # cropped_faces = []  # List to store the cropped faces

                for face_encoding in zip(face_encodings):
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding
                    )
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    if face_distances.size:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            accuracy = round(
                                (1 - face_distances[best_match_index]) * 100, 2
                            )
                            face_accuracies.append(accuracy)
                        else:
                            face_accuracies.append(None)
                    print(name)
                    face_names.append(name)

        return face_locs, face_names, face_accuracies, self.frame_resizing
