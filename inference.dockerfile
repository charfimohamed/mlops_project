FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt
COPY setup.py /code/setup.py
COPY src/ /code/src/
COPY src/models/checkpoints/ /code/checkpoints/
COPY ./requirements.txt /code/requirements.txt
COPY src/models/inference_images/ /code/inference_images/


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]