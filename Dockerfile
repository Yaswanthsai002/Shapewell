# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

<<<<<<< HEAD
#EXPOSE 5000
=======
EXPOSE 5000
>>>>>>> 6a2158f (First Commit)

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN apt-get update
<<<<<<< HEAD
RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install -y v4l-utils
RUN pip install protobuf==3.20
=======
RUN apt-get install ffmpeg libsm6 libxext6  libgl1 -y
RUN pip install protobuf==3.20.1
>>>>>>> 6a2158f (First Commit)

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
<<<<<<< HEAD
CMD ["python3", "app.py"]
=======
CMD ["python3","app.py"]
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
>>>>>>> 6a2158f (First Commit)