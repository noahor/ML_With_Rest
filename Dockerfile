FROM python:3.7-alpine
COPY . /app
WORKDIR /app

RUN apk add --no-cache --virtual .build-deps \
    build-base openssl-dev pkgconfig libffi-dev \
    cups-dev jpeg-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del .build-deps # delete the .build-deps in the same layer

RUN pip install -r requirements.txt
CMD python ./Hosting.py