FROM python:3.9-slim-buster
EXPOSE 5000
COPY inference /app
WORKDIR /app
RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]

