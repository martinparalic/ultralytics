# Ultralytics YOLO 🚀, AGPL-3.0 license
from shapely.geometry import Point, Polygon
from ultralytics.solutions.solutions import BaseSolution  # Import a parent class
from ultralytics.utils.plotting import Annotator, colors


class ZoneCounter(BaseSolution):
    """A class to manage the counting of objects in different zones using Ultralytics YOLO"""

    def __init__(self, **kwargs):
        """Initialization function for Count class, a child class of BaseSolution class, can be used for counting the objects."""
        super().__init__(**kwargs)
        self.zones = []
        self.counts = {}

    def count_objects_in_zones(self, im0):
        """
        Processes input data (frames or object tracks) and updates counts.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        if isinstance(self.region[0], tuple):
            self.zones.append(self.region[0])
            self.annotator.draw_region(
                reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
            )  # Draw
        elif isinstance(self.region[0], list):
            for z in self.region:
                self.zones.append(Polygon(z))
                self.annotator.draw_region(
                    reg_pts=z, color=(104, 0, 123), thickness=self.line_width * 2
                )  # Draw region

        self.counts = {index: 0 for index in range(len(self.region))}

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            bbox_center = (box[0] + box[2] / 2), (box[0] + box[2] / 2)
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.names[cls], color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width
            )

            # Check if detection inside region
            region_index = next((i for i, reg in enumerate(self.zones) if reg.contains(Point(bbox_center))), None)

            if region_index is not None:
                self.counts[region_index] += 1


        print(f"Region Counts : {self.counts}")

        # self.display_counts(im0)  # Display the counts on the frame
        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
