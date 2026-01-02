import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 720],
    [0, 720]
])

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("--webcam-resolution", nargs=2, type=int, default=[1280, 720])
    return parser.parse_args()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_resolution[1])

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    
    zone_polygon = ZONE_POLYGON * np.array(args.webcam_resolution).astype(int)
    zone = sv.PolygonZone(
        polygon=zone_polygon
    )
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE)

    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id != 0]
        
        labels = [
            f"{model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
        )
        
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels,
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        
        # Display the count
        count = zone.current_count
        cv2.putText(
            frame,
            f"Count: {count}",
            (10, 30),
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Frame", frame)
        if cv2.waitKey(30) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()