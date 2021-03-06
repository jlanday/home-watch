# home-watch

![alt text](https://github.com/jlanday/home-watch/blob/master/test.jpeg?raw=true)


Hi friends. home-watch is a docker image, designed to run on 64bit ARM hardware using a USB webcam.

Before going ahead please make sure you have Docker Buildx installed (https://docs.docker.com/buildx/working-with-buildx/)

To build 

```
docker buildx build --platform linux/arm64/v8 -t home-watch
```

To run 

```
docker run -d --privileged --restart=always --device=/dev/video0:/dev/video0 home-watch python3 app.py
```

This docker image has successfully ran on the following hardware:

```
Raspberry Pi 4
Nvidia Jetson Nano (2GB)
MacBook Pro 2014
```
To send 10K free a month emails:
```
https://console.cloud.google.com/marketplace/product/sendgrid-app/sendgrid-email
```

GLHF

(Special thanks to google, tensorflow, nvidia, docker, and all the other homies)
