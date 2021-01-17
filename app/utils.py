import base64
import datetime
import logging
import json
import time
import typing
import os

import cv2
from google.cloud import storage
import numpy as np
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from sendgrid.helpers.mail import (
    Mail,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
    ContentId,
)
import tensorflow as tf
import tensorflow_hub as hub

from plotters import *


class spy_cam:
    def __init__(
        self,
        bucket_name:str="justin-source-data",
        # See https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_object_detection.ipynb for other
        # compatible models.
        model_url:str="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        seconds_between_alerts:str="60",
        camera_name:str="dev",
        project_name:str="home-watch",
        gcp_creds_path:str="gcp_creds.json",
        sg_creds_path:str="sg_api_key.json",
        min_score_thresh:str="0.5",
        rotate_angle:str="180",
        adjust_color:bool=False,
        sg_from_email:str="justin@landay.rocks",
        sg_to_email:str="justinlanday@gmail.com",
        resolution:str="800,800",
        ignore_list:str="car=1.0",
        cache_size:str="5",
    ):
        self.bucket_name = bucket_name
        self.model_url = model_url
        self.seconds_between_alerts = int(seconds_between_alerts)
        self.camera_name = camera_name
        self.project_name = project_name
        self.gcp_creds_path = gcp_creds_path
        self.sg_creds_path = sg_creds_path
        self.min_score_thresh = float(min_score_thresh)
        self.rotate_angle = int(rotate_angle)
        self.adjust_color = adjust_color
        self.sg_from_email = sg_from_email
        self.sg_to_email = sg_to_email
        self.resolution = resolution.split(",")
        assert len(self.resolution) == 2, "resolution needs to be shape (dim1,dim2)"
        self.ignore_dict = {
            item.split("=")[0]: float(item.split("=")[1])
            for item in ignore_list.split(",")
        }

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.cache_size = int(cache_size)

        try:
            ch = self.logger.handlers[0]
            ch.setFormatter(
                logging.Formatter(
                    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s'",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.handlers[0] = ch
        except:
            ch = logging.StreamHandler()
            ch.setFormatter(
                logging.Formatter(
                    "[%(asctime)s][%(name)s][%(levelname)s] %(message)s'",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(ch)

        if self.gcp_creds_path:
            self.gcp_client = storage.Client.from_service_account_json(self.gcp_creds_path)
            self.gcp_bucket = self.gcp_client.bucket(bucket_name)

        if not os.path.exists("images"):
            os.mkdir("images")
            self.logger.info("made images")

        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, int(self.resolution[0]))
        self.cam.set(4, int(self.resolution[1]))
        self.logger.info("opened camera")
        self.model = hub.load(self.model_url)
        self.logger.info("loaded model")

        with open("category_index.json", "r") as f:
            self.category_index = json.loads(f.read())

        if self.sg_creds_path:
            with open(self.sg_creds_path, "r") as f:
                self.sg_api_key = json.loads(f.read())["sg_api_key"]

        self.mail_client = SendGridAPIClient(self.sg_api_key)
        self.cache = []

    def add_image_to_cache(self, image_path: str) -> None:
        """Appends an image_path to a cache for emailing"""
        self.cache.append(image_path)

    def clear_cache(self) -> None:
        """Clears the cache"""
        self.cache = []

    def send_email(self, subject: str = "Stranger Danger?", content: str = "<strong>:-)</strong>") -> None:
        """Emails all of the images in the cache"""

        # -- Initialize a template
        message = Mail(
            from_email=self.sg_from_email,
            to_emails=self.sg_to_email,
            subject=subject,
            html_content=content,
        )
        attachments = []
        # -- We want to reverse the cache so the images are sent in the order they are taken
        # -- We want to limit the number of images by the cache size.
        for index, image_path in enumerate(self.cache[::-1][: self.cache_size]):

            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                attachment = Attachment()
                attachment.file_content = FileContent(base64.b64encode(image_bytes).decode())
                attachment.file_type = FileType("image/jpg")
                attachment.file_name = FileName(image_path)
                attachment.disposition = Disposition("attachment")
                attachment.content_id = ContentId("ID{}".format(index))
                self.logger.info("added {}".format(image_path))
                message.add_attachment(attachment)
            except:
                self.logger.error("{} attatchment failure".format(image_path))

        response = self.mail_client.send(message)

        if response.status_code == 202:
            self.logger.info("email sent!")
        else:
            self.logger.error("email failure {}".format(email_response.body))

    def capture_image(self) -> (str, np.ndarray):
        """Uses a webcam to take an image and provides transformations"""

        # -- Construct image path
        now = datetime.datetime.now()
        dt = now.strftime("%Y-%m-%d")
        hms = now.strftime("%H:%M:%S:%f")
        if not os.path.exists("images/dt={dt}".format(dt=dt)):
            os.mkdir("images/dt={dt}".format(dt=dt))
        path = "images/dt={dt}/{batch}.jpg".format(dt=dt, batch=hms)

        # -- Take the photo
        _, image = self.cam.read()
        # -- Recall
        if self.adjust_color == "True":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # -- Rotate
        if self.rotate_angle:
            assert isinstance(self.rotate_angle, int), "rotate_angle needs to be an integer or None or False"
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, self.rotate_angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return path, image

    def batch_numpy_image(self, image: np.ndarray) -> np.ndarray:
        """Reshape to have batch size = 1"""
        return image.reshape((1, *image.shape)).astype(np.uint8)

    def make_predictions(self, image_np: np.ndarray) -> dict:
        """Runs image through arbitrary tensorhub model and returns results"""
        results = self.model(image_np)
        result = {key: value.numpy() for key, value in results.items()}
        return result

    def draw_boxes_on_predictions(self, image_np: np.ndarray, result: dict) -> np.ndarray:
        """Draws boundary boxes and labels on image"""
        image_np_with_detections = image_np.copy()

        # -- A helper function from google :)
        visualize_boxes_and_labels_on_image_array(
            image_np_with_detections[0],
            result["detection_boxes"][0],
            (result["detection_classes"][0]).astype(int),
            result["detection_scores"][0],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=self.min_score_thresh,
            agnostic_mode=False,
        )

        return image_np_with_detections[0]

    def disk_to_cloud(self, image_path: str):
        """Takes an image saved to disk and uploads it to all provided cloud storage vendors"""
        if self.gcp_client:
            try:
                gcp_path = self.disk_to_gcp(image_path)
                self.logger.info("image w boxes saved to cloud {}".format(gcp_path))
            except:
                self.logger.error("problem saving to gcp {}".format(gcp_path))
        # -- TODO: Implement other cloud cloud providers

    def disk_to_gcp(self, image_path: str) -> str:
        """Uploads a jpg to google cloud storage"""
        blob_path = "{project_name}/{camera_name}/{path}".format(
            project_name=self.project_name,
            camera_name=self.camera_name,
            path=image_path,
        )
        blob = self.gcp_bucket.blob(blob_path)
        blob.upload_from_filename(image_path, content_type="image/jpg")
        return blob_path

    def validate_results(self, results: dict) -> bool:
        """Checks your results against exclusions to determine notifications"""
        send_message = False
        class_indicies = np.where(results["detection_scores"][0] >= self.min_score_thresh)[0]
        class_labels = results["detection_classes"][0][class_indicies]
        class_labels = [
            self.category_index[str(int(index))]["name"] for index in class_labels
        ]
        class_scores = results["detection_scores"][0][class_indicies]
        for index, label in enumerate(class_labels):
            class_score = class_scores[index]
            message = "{}={:.2f}".format(label, class_score)
            self.logger.info(message)
            if label in self.ignore_dict.keys():
                if class_score >= self.ignore_dict[label]:
                    send_message = True
            elif class_score >= self.min_score_thresh:
                send_message = True
        return send_message
