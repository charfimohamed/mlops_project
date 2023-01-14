# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy over our application (the essential parts) from our computer to the container
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY src/models/checkpoints/ checkpoints/

# set the working directory in our container and add commands that install the dependencies
# --no-cache-dir tells pip to not use a cached version of the package from the local cache
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# the "u" here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal
ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]



