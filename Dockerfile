FROM python:3.11.9
LABEL authors="katagaki"

RUN pip install --upgrade pip
RUN apt update
RUN apt install -y ffmpeg cmake

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./models ./models
COPY ./*.py ./

CMD ["main.py"]