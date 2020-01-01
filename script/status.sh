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


APP_ID=$(ps -ef | grep gunicorn | grep ${API_NAME} | grep ${PORT} | awk '{print $2}')

if [ -n "$APP_ID" ]; then
    echo "================================ "
    echo "STATUS : Active"
    echo "PROCESS ID :" $APP_ID
    echo "================================ "
else
    echo "================================ "
    echo "STATUS : Not Active"
    echo "================================ "
fi
