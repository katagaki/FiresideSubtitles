FROM python:3.12.3
LABEL authors="katagaki"

RUN pip install --upgrade pip
RUN apt update
RUN apt --no-install-recommends install -y ffmpeg cmake

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./models ./models
COPY ./*.py ./

RUN groupadd -r firesiders -g 998
RUN useradd snake
RUN usermod -aG firesiders snake
USER snake

CMD ["python", "main.py"]