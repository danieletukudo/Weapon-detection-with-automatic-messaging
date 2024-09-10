import cv2
from ultralytics import YOLO
import yagmail


class WeaponVideoProcessor:
    """
    A class for processing video frames to detect weapons using YOLOv8 models.
    The class loads a YOLOv8 model for weapon detection, processes video frames,
    and applies annotations where weapons are identified. It also sends email alerts.
    """

    def __init__(self, model_path: str, email: str, email_password: str) -> None:
        """
        Initialize the WeaponVideoProcessor class with model path and email credentials.

        Args:
            model_path (str): Path to the YOLOv8 weights file.
            email (str): Email address to send notifications.
            email_password (str): App password or authorization token for the email.
        """
        self.model_path = model_path
        self.email = email
        self.email_password = email_password
        self.model = YOLO(model_path)

    def send_email(self, to_email: str, object_detected: str) -> None:
        """
        Sends an email alert when a weapon is detected.

        Args:
            to_email (str): The recipient email address.
            object_detected (str): The label of the detected object (e.g., weapon).
        """
        # Initialize yagmail client (this will prompt you for password the first time)
        yag = yagmail.SMTP("danielsamueletukudo@gmail.com", "snslmilczpcsfmcm")

        # Send email
        yag.send(
            to=to_email,
            subject="Security Alert",
            contents=f'ALERT - {object_detected} objects has been detected!!'
        )

        print("Email sent successfully!")

    def process_video(self, video_source: str) -> None:
        """
        Processes a video stream to detect weapons and sends an email alert when detected.
        The detected weapons are annotated on the frames, and the video is displayed.

        Args:
            video_source (str): Path to the video file or '0' for webcam.

        """
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}.")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from video.")
                break

            results = self.model.track(frame, persist=True, verbose=True)
            detected_objects = []

            if results[0].boxes:  # Check if there are any boxes detected
                boxes = results[0].boxes.xyxy.cpu()
                class_ids = results[0].boxes.cls.cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()
                names = self.model.names

                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    if confidence >= 0.5:
                        detected_object = names[int(class_id)]
                        detected_objects.append(detected_object)

                        label = f"{detected_object} {confidence:.2f} "

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        # y1 = max(y1, label_size[1] + 100)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1 + base_line - 10),
                                      (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        text = " Weapon Detected and Email Sent Successfully"

                        cv2.putText(frame, text, (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        if True:
                            pass
                            self.send_email("danielsamueletukudo@gmail.com", label)

            # Show the processed video
            cv2.imshow("Weapon Detection Output", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration
    model_path = "best (2).pt"  # Update with your model path
    email = "danielsamueletukudo@gmail.com"
    email_password = "snslmilczpcsfmcm"  # Replace with your actual app password

    # Initialize the processor and process the video
    video_source = 0  # Replace with your video file path or '0' for webcam
    processor = WeaponVideoProcessor(model_path, email, email_password)
    processor.process_video(video_source=video_source)
