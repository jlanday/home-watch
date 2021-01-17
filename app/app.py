def main(config):
    camera = spy_cam(
        bucket_name=config.get("bucket_name"),
        model_url=config.get("model_url"),
        seconds_between_alerts=config.get("seconds_between_alerts"),
        camera_name=config.get("camera_name"),
        project_name=config.get("project_name"),
        gcp_creds_path=config.get("gcp_creds_path"),
        sg_creds_path=config.get("sg_creds_path"),
        min_score_thresh=config.get("min_score_thresh"),
        rotate_angle=config.get("rotate_angle"),
        sg_from_email=config.get("sg_from_email"),
        sg_to_email=config.get("sg_to_email"),
        resolution=config.get("resolution"),
        adjust_color=config.get("adjust_color"),
        ignore_list=config.get("ignore_list"),
        cache_size=config.get("cache_size"),
    )

    t0 = time.time()
    while True:
        image_path, image = camera.capture_image()
        camera.logger.info("image captured.")

        image_np = camera.batch_numpy_image(image)
        camera.logger.info("image loaded into numpy")

        results = camera.make_predictions(image_np)
        camera.logger.info("image made preditions")
        # -- We want to ignore results we're not interested in
        valid_results = camera.validate_results(results)
        if valid_results:
            camera.logger.info("Valid Results")
            # -- Saves path to cache
            camera.add_image_to_cache(image_path)
            # -- Draws boxes & labels
            image_w_boxes = camera.draw_boxes_on_predictions(image_np, results)
            # -- Saves images to disk
            im = Image.fromarray(image_w_boxes)
            im.save(image_path)
            camera.logger.info("image w boxes saved to {}".format(image_path))
            # -- Save to GCP
            camera.disk_to_cloud(image_path)
        else:
            camera.logger.info("No entities detected")

        seconds_elapsed = int(time.time() - t0)
        if seconds_elapsed >= camera.seconds_between_alerts and len(camera.cache) > 0:
            email_response = camera.send_email()
            camera.clear_cache()
            t0 = time.time()


if __name__ == "__main__":
    from utils import *

    with open("app_config.json", "r") as f:
        config = json.loads(f.read())

    main(config)
