# Docker 

Docker is a platform designed to make it easier to develop, deploy, and run applications by using containers. A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings. They isolate software from its environment and ensure that it works uniformly despite differences for instance between development and staging.

A few key concepts related to Docker Containers:
1) ``Docker Engine``

    It is the underlying software that manages containers on a host system and includes a server, a REST API, and a command-line interface (CLI) for interacting with containers.

2) ``Dockerfile``

    A script that contains a set of instructions for building a Docker image. They provide a declarative syntax to define the steps and components required to create a Docker image 

3) ``Docker Image``

    A lightweight, standalone, executable package that includes everything needed to run a piece of software, including the code, a runtime, libraries, environment variables, and config files. Images are used to create containers.

4) ``Docker Container``

    A running instance of a Docker image. It encapsulates an application and its dependencies, isolating it from the host system and other containers. Containers are portable, consistent, and can be easily moved between different environments.

It is important to note containers virtualize the operating system instead of the hardware, but share the operating system kernel. While Docker supports Linux and Windows based applications, it can only run an operating system that shares the same kernel e.g, Debian, Fedora, CentOS share the Linux kernel but Windows does not.

![High Level Architecture](<../assets/Docker Containers High level view.png>)

## Installation

As mentioned previosuly, Docker Engine is an open-source containerization technology for building and containerizing applications. It acts as a client-server application with a server, APIs, and a commandline interface (CLI). The CLI uses Docker APIs to control or interact with the Docker daemon through scripting or direct CLI commands. Many other Docker applications use the underlying API and CLI. The daemon creates and manage Docker objects, such as images, containers, networks, and volumes.

To [install Docker Engine](https://docs.docker.com/engine/install/) for your platform, Docker provides several paths to do this. Docker Engine is available on a variety of platforms such as Linux distros, MAC OS, Windows, etc. Simply choose the required platform and clik on the associated link to proceed forward. Since I have a Windows OS native to my hardware, I gave subsequently chosen to install Docker Engine shipped as [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) for Windows. The installation process is fairly simple as outlined below:

- Check underlying system requirements and choose one of the two supported methods:

    1) ``WSL-2 Backend`` 
        
        With WSL-2, we have a compatibility layer for running linux binary executables on Windows i.e., Linux Kernel   alongside the Windows Kernel. This approach dicates that the OS kernel is shared between all containers.

    2) ``Hyper-V Backend``

        With this approach, one has complete Kernel isolation between conatiners. 

        Important to note, both approaches can not be implemented at the same time on the same host. Once you choose your preferred approach based on the use-case and setup, follow the instructions outlined to enable services and meet prerequistes.

- Install Docker Desktop on Windows

    You can do this interactively using the installation wizard or through the command line interface. Follow the instructions carefully to ensure successful installation.

- Start Docker Desktop

    Docker Desktop does not start automatically after installatiion. Go to the search bar and select Docker Desktop, accept the terms and you are good to go.

![Docker Desktop](<../assets/Docker Desktop.png>)

## Workflow

Building a Docker image involves creating a portable and executable package that includes all the dependencies and configurations needed to run an application. Here's a step-by-step process for building a Docker image and then running a container based on that image:

![Docker Workflow](<../assets/Docker workflow.png>)

### Create a Dockerfile

Create a file named ``Dockerfile`` in the root directory of your project. The Dockerfile is a text file that contains instructions to build the Docker image. It typically includes commands to set up the environment, copy application code, install dependencies, and define runtime settings. Note: You can have multiple Dockerfiles with different names for the same application to build different version of the image in the root directory.

Here is a simple example of a Dockerfile that is used to create a python application:

```
# Use an official Python runtime as a base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "app.py"]
``` 

### Build Image

Once you have populated the Dockerfile, open a terminal, navigate to the directory containing your Dockerfile, and run the following command to build the Docker image:

```
docker build . -t image_name
```
This tells the docker deamon to start building the image on the host. The ``-t`` flag specifies the name for the image. You can replace 'image_name' with anything e.g., 'smat' for our application. Note: You can have multiple version of the same image by specifying ``tags`` for the image. This is done when there are slight changes or updates to the application code for example.

```
docker build . -t image_name:tag
```
Important to understand that Docker Engine builds images in a ``Layered  Architecture``. This means that each line of instruction creates a new layer that reflects chnages form the previous layer. To save disk space on the local host, Docker caches layers when rebuilding images using the local filesystem. Additionally, once the build process is complete and a Docker Image is created, one cannot change it.

### Push Image (Optional)

You can push your image to a container registry, making it available for others to use and for deployment on other systems. [Docker Hub](https://hub.docker.com/) is a popular public registry, but there are other private or enterprise registries you can use. First, you need to login to Docker Hub using your credentials. This can be done using the following command:

```
docker login
```
Once logged in, tag your image appropriately before commencing to push it:

```
docker push your-docker-hub-username/image_name:tag
```
You can go to the [Docker Hub website](https://hub.docker.com/) and check your account to verify that the image has been successfully pushed.

Additionally, you can pull images from repositories like Docker Hub onto your host. This can be done in a simple step as outlined below:

```
docker pull your-docker-hub-username/image_name:tag
```
This downloads a local version of the image on your platform, ready to be run as a container.

### Run Container

The final step in the workflow. After successfully building the Docker Image, you can run a container based on that image. To run a container in a non-interactive or detached manner, use the following commands:

```
docker run image_name:tag

OR

docker run -d image_name:tag
```

Alternatively, you can run the container in an interactive mode and access the pseudo terminal by simply specifying the ``-it`` parameters/flags:

```
docker run -it image_name:tag
```
This allows you to perform actions within the container environment, run processes, make changes etc. New files created in the container are stored in the read-write (container layer), that sits on top of the image layers.

However, it is important to note that the container is an isloated filesystem, so changes made inside it no longer exists once the conatiner stops. If one wishes to persist data i.e., data remains after the container has stopped, we can do so by mounting/mapping to a directory outside the Docker container. There are two approaches to it:

- ``Volume Mount``
    
    Volumes are created and managed by Docker. When you create a volume, it is stored within a directory on the Docker host. When you mount the volume into a container, this directory is what is mounted into the container. They are managed by Docker and are isolated from the core functionality of the host machine. A given volume can be mounted into multiple containers simultaneously. You ca create a volume explicitly, or Docker can create a volume during container or service creation.
    ```
    docker volume create

    OR 

    docker run -v volume:/path_in_container image_name:tag 
    ```

- ``Bind Mount``

    When you use a bind mount, a file or directory on the host machine is mounted into a container. The file or directory is referenced by its full path on the host machine. The file or directory doesn't need to exist on the Docker host already. It is created on demand if it does not yet exist. Bind mounts are very performant, but they rely on the host machine's filesystem having a specific directory structure available. 
    ```
    docker run --mount type=bind,source=path_on_host_machine,target=path_in_container image_name:tag
    ```

## Singularity

[Singularity](https://docs.sylabs.io/guides/3.5/user-guide/quick_start.html) is a container platform. It allows you to create and run containers that package up pieces of software in a way that is portable and reproducible. It was created to run complex applications on HPC clusters in a simple, portable, and reproducible way.

Some salient features attributed to ``Singularity``:

- Verifiable reproducibility and security, using cryptographic signatures, an immutable container image format, and in-memory decryption.
- Integration over isolation by default. Easily make use of GPUs, high speed networks, parallel filesystems on a cluster or server by default.
- Mobility of compute. The single file SIF container format is easy to transport and share.
- A simple, effective security model. You are the same user inside a container as outside, and cannot gain additional privilege on the host system by default. 

Since ``SPEED`` by default supports Singularity and not Docker, we make use of their interoperability for running containers. Singularity uses its own file format called the ``SIF``. You can either use the ``build`` command to create images from other images or from scratch using a definition file. You can also use  ``build`` to convert an image between the container formats supported by Singularity. For both methods, you need to provide the `URI`

To ``pull`` or ``build`` images from an external repository such as the Docker Hub, please apply either of the following commands:

```
singularity pull docker://docker_hub_username/image_name:tag

OR 

singularity build container_name.sif docker://docker_hub_username/image_name:tag
```
To run the container as if it was an executable in an non-interactive manner, please use the `run` command in the following manner:

```
singularity run container_name.sif
```

To interact with the container, use the `shell` command to spawn a new shell within your container and interact with it as though it were a small virtual machine.

```
singularity shell container_name.sif
```

## SMAT-Docker Workflow

This section outlines how to pull Docker Image(s) for our codebase and then run the containers on your individual platform or on SPEED using Singularity.

1) ``Standard Platform``

    Please follow the steps below:

    Navigate to the [github repository](https://github.com/goutamyg/SMAT) and download the Dockerfile(s) based on your preference. Open a terminal in the directory where the Dockerfile(s) are stored.

    ```
    docker build . -f Dockerfile/Dockerfile_With_Test_Data -t smat:latest/got10k
    ```

    This steps builds the Docker image locally on your host machine and caches it using the native filesystem. Choose one Dockerfile at a time. However, if you wish to pull the image without building it first, follow the steps below:

    ```
    docker pull hassankhan17/smat:latest
    ```
    This pulls the latest or deafult Image version to your host machine. Note: The default version does not include datasets i.e., got10k, lasot, etc, so one has to follow the steps outlined in the official github repository from the tracker evaluation section onwards. 

    Alternatively, if you wish to avoid downloading the tracker dataset separately i.e., got10k, you can pull the other image version.
    ```
    docker pull hassankhan17/smat:got10k
    ```
    You can simply perform inference without having to create the dataset directory and downloading datasets from online sources or copying from local storage.

    Once the image is successfully pulled from Docker Hub, simply run it in the interactive mode by secifying the appropriate parameters/flags.

    ```
    docker run -it smat:latest

    OR 

    docker run -it smat:got10k
    ```

    This option does not mount a volume or bind. Thus, if you wish to persist your data once the container has stopped running, execute one of the two commnds:

    ```
    docker run -it -v volume:/tmp/volume smat:latest

    OR

    docker run -it --mount type=bind,source=path_to_local_directory,target=/tmp/data smat:latest
    ```

    You can essentially run the container with Visual Studio Code thus allowing for interactive usage.

2) ``SPEED Platform``

    We cannot directly work with Docker on SPEED, therefore first load Singularity module with the following command:

    ```
    module load singularity/default
    ```
    Once it is loaded, pull the Docker Image(s) from Docker Hub or run the image directly:

    ```
    singularity pull docker://hassankhan17/smat:speed
    ```
    Give some time for the image to be pulled. Once the process concludes successfully, the image is downloaded from Docker Hub and cached locally, thus not requiring re-download for subsequent container runs.
    
    We may proceed ahead by running the container using the shell command for interactive session (preferred method):
    
    ```
    singularity shell docker://hassankhan17/smat:speed
    ```