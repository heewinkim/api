#!/bin/bash

printf "API NAME [defualt:fd] :"
read API_NAME
if [ -n ${API_NAME} ];then
    API_NAME=fd
fi
printf "PORT [defualt:5000] :"
read PORT
if [ -n ${PORT} ];then
    PORT=5000
fi
printf "WORKER [defualt:1] :"
read WORKER
if [ -n ${WORKER} ];then
    WORKER=1
fi

#export LC_ALL=C
#export PATH="$HOME/.pyenv/bin:$PATH"
#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"
#pyenv activate face-detection

echo "API NAME : $API_NAME"
echo "PORT: $PORT"
echo "WORKER: $WORKER"

nohup gunicorn --bind 0.0.0.0:$PORT src.main -w ${WORKER} --log-file gunicorn.log --timeout 60 --name ${API_NAME} >> gunicorn.log 2>&1 &

sleep 2

APP_ID=$(ps -ef | grep gunicorn | grep ${API_NAME} | grep ${PORT} | awk '{print $2}')

if [ -n "$APP_ID" ]; then
    echo "================================ "
    echo "STATUS : Active"
    echo "PROCESS ID :" $APP_ID
    echo "================================ "
else
    echo "================================ "
    echo "STATUS : Not Active"
    echo "PROCESS ID :" $APP_ID
    echo "================================ "
fi