class OnlineDetector:
    '''
    Please Download saved model from google disk
    https://drive.google.com/file/d/1K1j5U1kCkmRb19FhTPyx4yo8nEXgxJPK/view?usp=sharing
    and unzip it to EXPORTED_MODEL_FOLDER. Change link to folder in code below

    Please Download annotations from google disk
    https://drive.google.com/file/d/1TlFn9XzVLk5vEsBskitML3OOq6025iLq/view?usp=sharing
    and unzip it to ANNOTATIONS_FOLDER. Change link to folder in code below

    Make shure to turn on your camera.
    To turn off camera press 'q'
    '''
    import tensorflow as tf
    from object_detection.utils import config_util
    import os
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    import cv2
    import numpy as np

    EXPORTED_MODEL_FOLDER='playing_cards\\exported_models\\'
    ANNOTATIONS_FOLDER='playing_cards\\annotations\\'

    def __init__(self):
        # Load pipeline config and build a detection model
        configs = self.config_util.get_configs_from_pipeline_file(
            self.EXPORTED_MODEL_FOLDER + 'pipeline.config')
        self.detection_model = self.model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = self.tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(
            self.os.path.join(self.EXPORTED_MODEL_FOLDER + 'checkpoint\\',
                         'ckpt-0')).expect_partial()

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def run_camera(self):

        category_index = self.label_map_util.create_category_index_from_labelmap(
            self.ANNOTATIONS_FOLDER + 'label_map.pbtxt')

        # Setup capture
        cap = self.cv2.VideoCapture(0)
        width = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            image_np = self.np.array(frame)

            input_tensor = self.tf.convert_to_tensor(self.np.expand_dims(image_np, 0), dtype=self.tf.float32)
            detections = self.detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(self.np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            self.viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

            self.cv2.imshow('object detection', self.cv2.resize(image_np_with_detections, (800, 600)))

            if self.cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break

        detections = self.detect_fn(input_tensor)

if __name__ == '__main__':
    watch = OnlineDetector()
    watch.run_camera()