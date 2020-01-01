from common.framework import PyFlask
from common.core import PyImage
from common.core.error import *
from src.mtcnn import FaceDetector
application = PyFlask('fd')
face_detector = FaceDetector()


@application.route('/v1/heewinkim/fd',methods=['POST'])
def request_fd():

    application.output.set_default(keys=['request','response'])
    if application.json:

        # pre request check
        data = application.json
        application.validate_keys(data,['imageData','types'],True)

        # request data add
        application.output.set_output({'reqeust':{**data}})

        # preprocessing
        img_cv = PyImage.preprocessing_image(data['imageData'],data['types'],data.get('ot',0),'cv2')

        # core processing
        result = face_detector.run(img_cv)

        # post processing
        return application.output.return_output({'response':{**result}})
    else:
        raise PyError(ERROR_TYPES.REQUEST_ERROR,'Not json data offered.')

if __name__ == '__main__':

    application.run()