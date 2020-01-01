#### branch details

- fd : mtcnn(mxnet)
- submodule : heewinkim/python-utils

#### how to serving using gunicorn

```
# ${PWD}==${PROJECT_ROOT}
# pyenv activate ${PYENV_NAME} # PYTHON_VERSION==3.6.1
# pip install -r requirements.txt
```
./script start.sh
./script status.sh
./script stop.sh

#### how to test

$ curl -X POST -H Content-Type:application/json -d "@sample.json" http://0.0.0.0:5000/v1/heewinkim/fd
