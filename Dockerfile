FROM python:3.10-slim

MAINTAINER <ershixiong> king101125s@gmail.com

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

EXPOSE 7860

CMD ["python", "web.py"]

# $ docker build -t mobot .
# $ docker run --name mobot_1 -p 7860:7860 -d mobot:latest