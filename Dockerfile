FROM python:3.7.12

WORKDIR /SNHU

COPY requirements.txt  .

RUN pip install -r requirements.txt

COPY .  .

CMD ["python","./SNHU/v1/utils/main.py"]

