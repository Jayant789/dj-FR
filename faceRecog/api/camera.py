import cv2
from .simple_facerec import SimpleFacerec  # Import your face recognition logic here
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import time


class VideoCamera(object):
    def __init__(self):
        # Initialize Firebase
        cred = credentials.Certificate(
            {
                "type": "service_account",
                "project_id": "abl-security-b033d",
                "private_key_id": "ed63ecb605fa96b95c54d642033f71de592f6c97",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC0zYE5H0vdeBPG\nalokzev5DD8MtRltEwGyZLNCDnlXdYyBNYsbzMc0kEyJ6+c3cZIsyFLWVdQBGdJt\noy15RRLEPw1nPY6r025dToRCfxbdrtGo4bg9mOF4CB4jLLyfFjKmvBh1JqGVIpMl\n4RZrH0O7SS226ldtl3EPdd5BPCnsL4CsatnEd89jNvIpShYKjxvoeHVNGxOKnuAG\n69lWGx2Kj2E/qr26WtfJC0CfHQEICIKbLHPDwqkK4xkdFrBoJtKNyWZM9b7MDZ8D\nKr0aGIYQ69D+/WfM47We1+XARNkBlZH/KoyqGEqJYZxljKKFtzOYkvmMKhnaMgwG\nr4JLPOtzAgMBAAECggEAArKri99TON2WQkZ7zFAJmsb1JQRWt3NqYD8e4EyQofg1\nS9SxXn4rMhdXdlKHWKL+J0RRvq94/BmzXJJhwm5mkv041fWmnTlqCgG4ReH9Fp7/\nYY8PbZLCECaRQnrUo2XiqNi4X4KDNE40CG73/6bGuL5ii07d71Cxiy+dx5gu5VMG\nrJqKGfgN2Fks28O056deaShpP59DMJmd+dKkyISOMVNKYxEHs08PL4bCUuFKyo0m\n8unJZ1Xax+dyxO3bFubBKGdBtBHCwlVmJXY9+MMYkgO/fOLoLQhjfFxYiWyb2jeE\nCbrwLuiz/wv6ZxZQ+tO9vgk2hIlL08UD7wet9PetkQKBgQD8+9Z/6yJjfjDKkzcy\n+y3nYc11FX4ss3nlyhDOTmf4TIQWmtRfkXQp9g6b9VNXJyCsGce4wOim7wNBtyZE\nj5eNBzHLZH2QZvv+qgRbVAU31MytOBXwZTxdgmOlvKrel2RP5JYiMVU26zopJ5uz\nRlVg8IpyVJK2i5yfurKkJyTBEQKBgQC29VrHtq+0Mjio4Hlb3SA3PdTN+qyRhFKE\nUuxhpqT/FfszHciHUPfpE/f3G1dquiFzAOJYuCrvLCTWJHEYKvHBtlYMueYNFTOt\nk/6kFd4ZZ+q/XJVGF4NFofTIN9L2UbWHc52VwG3WDmY7U1K7VjVoXw/tdsUJB6Gv\nBuxCSfgkQwKBgQDiSW2jfDKFZjHEcYw1aOG1jxEVQsVavKszdNw1fYKYYfDgu1tt\npJCQnAyTSgxi75fU+TZhtwQjlbWHCYkMWJiJyD6tHNUH3mZXc8Jz4qLMPudZpcpR\n/mvRhLkXXbxFYIuUvvXf3drIRf3/I/OslyP1kxNzktysthLB+WCjXnQM0QKBgQCS\nF2rMrECyx6Ncnhnp07FEyxeg/jhL3fgx9zEPbIy1r2ytTWvxOSMsNyi6ZVexPj01\nYpBavXxzDLHBWMoBvVDcGGevs8VRzws74D/l8Bwv9z2IXjpIBMBqmr8mHQVUcLxe\nE2DS0hwiX88cMhWOx3DQDZBfUoZVBoYBh6qh6AS/lwKBgClU7MFAZZIwLTc/633e\neJFT+wWIyz8GhikCPQ7rnSI81Goeklr2223Rtm5wd86HYqnHKuSJiuwbc77TAzQC\nnL0UFFjxfN2G9qPOC4ZcvUWtnpYfJrH9e3mwX5pm1YgYeIGBoPTPVU12FIXo7A1f\nEKVE0T7BdBBQ9HCnSC8MDfOv\n-----END PRIVATE KEY-----\n",
                "client_email": "firebase-adminsdk-6rtr5@abl-security-b033d.iam.gserviceaccount.com",
                "client_id": "106922814617125507892",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-6rtr5%40abl-security-b033d.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com",
            }
        )
        firebase_admin.initialize_app(cred)

        self.db = firestore.client()

        # Initialize SimpleFacerec or your face recognition logic here...
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images(
            "C:/Users/jayan/Desktop/Argenbright/dj-FR/faceRecog/api/images"
        )  # Load your face encodings
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.last_data_time = time.time()

    def __del__(self):
        self.cap.release()

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            face_locations, face_names = self.sfr.detect_known_faces(frame)

            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                if name == "Unknown":
                    # print(name + "if")
                    cv2.putText(
                        frame,
                        name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 200, 0),
                        2,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)

                else:
                    print(name + "else")
                    cv2.putText(
                        frame,
                        name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 0, 200),
                        2,
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                    current_time = time.time()
                    if current_time - self.last_data_time >= 10:
                        # Prepare the document data and save it to your Firestore database
                        document_data = {
                            "Name": name,
                            "Date": datetime.now().date().strftime("%Y-%m-%d"),
                            "Time": datetime.now().time().strftime("%H:%M:%S"),
                        }

                        # Use a formatted timestamp as the document ID
                        timestamp_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Add the document to the "suspects" collection using the timestamp as the ID
                        self.db.collection("Suspects").document(timestamp_id).set(
                            document_data
                        )
                        print(name + " Updated in Database")

                        # Update the last_data_time
                        self.last_data_time = current_time

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                break

            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
