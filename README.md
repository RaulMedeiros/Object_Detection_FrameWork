<p align="center">
  <img src="http://lapisco.ifce.edu.br/wp-content/uploads/2018/03/cropped-LOGO-06.png" width="400" alt="accessibility text">
</p>

# Object_Detection_FrameWork


## Install Docker
``` shell
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce
```
``` shell
# or (Not Tested Yet)
$ curl -fsSL http://get.docker.com | bash

```

### Teste Docker (Optional)
``` shell
$ sudo docker run hello-world

```
## Build and Run Object_Detection_FrameWork Container Image
``` shell
$ sudo docker build -t tensorflow_obj_detect .
$ sudo docker run -it --net host --device /dev/video0 tensorflow_obj_detect
```
## Run Object_Detection_FrameWork Application
``` shell
$ python obj_detect_stream_server.py
```
Or if you decide to use `docker-compose.yml`, you can build and run in a single command as following:  (Not Tested Yet)

``` shell
$ sudo docker-compose up
```
 
### Server Arguments
| Command | Default |
| --- | --- |
| --host| localhost |
| --port| 8080 |

### CV_APP Arguments
| Command | Default |
| --- | --- |
| --source | 0 |
| --decoder	 | /tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt |
| --model | ssd_mobilenet_v2_coco_2018_03_29 |

#### Available Models 
|  id |  --model |
| --- | --- |
| 1 | ssd_mobilenet_v1_coco_2018_01_28
| 2 | ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
| 3 | ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18
| 4 | ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18
| 5 | ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
| 6 | ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
| 7 | ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
| 8 | ssd_mobilenet_v2_coco_2018_03_29
| 9 | ssdlite_mobilenet_v2_coco_2018_05_09
| 10 | ssd_inception_v2_coco_2018_01_28
| 11 | faster_rcnn_inception_v2_coco_2018_01_28
| 12 | faster_rcnn_resnet50_coco_2018_01_28
| 13 | faster_rcnn_resnet50_lowproposals_coco_2018_01_28
| 14 |rfcn_resnet101_coco_2018_01_28
| 15 |faster_rcnn_resnet101_coco_2018_01_28
| 16 |faster_rcnn_resnet101_lowproposals_coco_2018_01_28
| 17 |faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
| 18 |faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28
| 19 |faster_rcnn_nas_coco_2018_01_28
| 20 |faster_rcnn_nas_lowproposals_coco_2018_01_28
| 21 |mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
| 22 |mask_rcnn_inception_v2_coco_2018_01_28
| 23 |mask_rcnn_resnet101_atrous_coco_2018_01_28
| 24 |mask_rcnn_resnet50_atrous_coco_2018_01_28

##
http://localhost:8080/ (It may take a while...)
