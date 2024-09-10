import cv2
from ultralytics import YOLO
import os
import yagmail


class WeaponImageProcessor:
    """
    A class for processing images to detect weapons using YOLOv8 models.
    The class loads a YOLOv8 model for weapon detection, processes images,
    and applies blurring to detected regions where weapons are identified.
    """

    def __init__(self) -> None:
        """
        Initialize the WeaponImageProcessor class with default email settings.
        """
        self.to_email = "danielsamueletukudo@gmail.com"

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

    def load_yolo_model(self, model_path: os.path) -> YOLO:
        """
        Load the YOLOv8 model for weapon detection.

        Args:
            model_path (os.path): Path to the YOLOv8 weights file.

        Returns:
            YOLO: An instance of the YOLO model loaded with the specified weights.
        """
        self.model = YOLO(model_path)
        return self.model

    def process_image(self, image_path: str) -> None:
        """
        Detect weapons in an image and blur the detected weapon regions.

        Args:
            image_path (str): Path to the image file.
            blur_ratio (int): Ratio for blurring weapon regions.
        """
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not read image {image_path}.")
            return

        model = self.load_yolo_model("best (2).pt")

        results = model.track(frame, persist=True, verbose=True)
        if results[0].boxes:  # Check if there are any boxes detected
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()
            names = model.names

            if boxes is not None:
                for box, cls, score in zip(boxes, clss, scores):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)

                    if score >= 0.5:
                        detected_object = model.names[int(cls)]

                        label = f"{detected_object} {score:.2f} "

                        # Draw bounding box and label on the image
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1 + base_line - 10),
                                      (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        text = "Weapon Detected and Email Sent Successfully"
                        cv2.putText(frame, text, (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        # Send email notification
                        self.send_email(self.to_email, label)

        # Display the processed image
        cv2.imshow("Weapon Detection Output", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    image_file = 'image.png'  # Replace with the path to the image
    processor = WeaponImageProcessor()
    processor.process_image(image_path=image_file)
