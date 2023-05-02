# Building Custom Images

## Requirements

To use custom images to run a workspace, your custom images have to satisfy below requirements.

* Jupyterlab
  * VESSL runs Jupyterlab and expose port `8888`. Jupyterlab should be pre-installed in the container image.&#x20;
  * Jupyterlab daemon must be located in `/usr/local/bin/jupyter`.
* sshd
  * VESSL runs sshd and expose port `22` as NodePort. sshd package should be pre-installed in the container image.
* PVC mountable at `/home/vessl`
  * VESSL mounts a PVC at `/home/vessl` to keep state across Pod restarts.

## Building from VESSL's pre-built images

VESSL offers pre-built images to run workspaces directly. You can use these images to build your own images. These images already have pre-installed Jupyterlab and sshd. Currently we offer 6 pre-built images. You can get these images from AWS ECR.

| Python Version | CUDA Version | Image URL                                           |
| -------------- | ------------ | --------------------------------------------------- |
| 3.6.14         | CPU Only     | public.ecr.aws/vessl/kernels:py36.full-cpu          |
| 3.6.14         | 10.2         | public.ecr.aws/vessl/kernels:py36-cuda10.2.full-gpu |
| 3.6.14         | 11.2         | public.ecr.aws/vessl/kernels:py36-cuda11.2.full-gpu |
| 3.7.11         | CPU Only     | public.ecr.aws/vessl/kernels:py37.full-cpu          |
| 3.7.11         | 10.2         | public.ecr.aws/vessl/kernels:py37-cuda10.2.full-gpu |
| 3.7.11         | 11.2         | public.ecr.aws/vessl/kernels:py37-cuda11.2.full-gpu |

### Example

```
# Use Python 3.7.11, CUDA 11.2 image
FROM public.ecr.aws/vessl/kernels:py37-cuda11.2.full-gpu.jupyter

# Install Python dependencies
RUN pip install transformers
...
```

## Building from community maintained images

You can make your own images from any community maintained Docker images. But you have to be sure that your image meet our requirements.

### Example

```
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04
ENV PYTHON_VERSION=3.7.11

RUN apt-get update
# Install sshd
RUN apt-get install -y -q openssh-server

# Install dependencies for installing Python
RUN apt-get install -y -q \
    wget \
    zlib1g-dev \
    openssh-server \
    curl \
    libssl-dev \
    libffi-dev

# Install dependencies for Jupyterlab
RUN apt-get install -y -q \
    libsqlite3-dev

# Install Python
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
RUN tar -xvf Python-$PYTHON_VERSION.tgz
RUN /bin/bash -c "cd Python-$PYTHON_VERSION/; ./configure; make install"
RUN rm -rf Python-$PYTHON_VERSION*
RUN update-alternatives --install /usr/bin/python python $(which python3) 1

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python && pip install -U pip

# Install Jupyterlab
RUN pip install jupyterlab
```
