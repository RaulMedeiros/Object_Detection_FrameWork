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
$ sudo docker run --net host -i -t tensorflow_obj_detect
```
## Run Object_Detection_FrameWork Application
``` shell
$ python object_detection_api.py
```

