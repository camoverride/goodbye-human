import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import numpy as np



def convolve(masks, background, scene):
    """
    Takes segement masks, combines them to one mask, and then applies the `background`
    on top of the `scene` in the area covered by the `masks`.

    Parameters
    ----------
    masks : np.ndarray
        An array of shape (h, w, n) where h and w are the image height/width
        `n` corresponds to the number of masks.
        All values are boolean True or False.
    background : np.ndarray
        An array of shape (h, w, 3) that represents a picture of a
        static background that should infrequently change.
    scene : np.ndarray
        An array of shape (h, w, 3) that represents the scene captured
        by the webcam, where things might have changed.

    Returns
    -------
    np.ndarray
        An array where masks have been applied to erase certain objects.
    """
    # Combine all masks.
    num_masks = masks.shape[-1]
    single_mask_shape = [masks.shape[-3], masks.shape[-2], 1]

    all_masks = masks[:,:,0].reshape(single_mask_shape)
    if num_masks > 1:
        for i in range(0, num_masks):
            all_masks += masks[:,:,i].reshape(single_mask_shape)
    all_masks = all_masks.reshape(single_mask_shape)
    
    # Select all the pixels in the scene except for those covered by the mask.
    scene_minus_mask = np.invert(all_masks) * scene

    # Select the pixels from the background covered by the mask.
    mask_filled_with_background = all_masks * background

    # Combine the two to get a scene with humans erased.
    scene_with_erased_human = scene_minus_mask + mask_filled_with_background

    return scene_with_erased_human


def get_background():
    """
    Gets a background image. Assumes a static camera.
    """
    background = None
    for _ in range(5): # First image might fail.
        vid = cv2.VideoCapture(0)
        _, frame = vid.read()
        background = frame
    return background


def video_stream(model, target_classes, background, camera):
    """
    Starts a videostream where certain objects have been erased.

    Parameters
    ----------
    model : instanceSegmentation
        A model that will be used for detection.
    target_classes : object
        Classes to be detected by the model (and subsequently deleted from the image).
    background : np.ndarray
        An array of dimensions (h, w, 3) that represents a background that should.
        change infrequently. Assumes a static camera.

    camera : int or str
        A local camera (0) or RTSP stream (some string).

    Returns
    -------
    None
        Streams a video.
    """
    # Video boilerplate
    vid = cv2.VideoCapture(camera)

    # Main event loop.
    while(True): 
        _, frame = vid.read()

        # Perform object detection. Masks overlay objects that are detected.
        segmask_data, _ = model.segmentFrame(frame, segment_target_classes=target_classes)
        masks = segmask_data["masks"]

        # Generate a new image where the detected objects are erased, if there are any.
        if masks.any():
            image = convolve(masks=masks, background=background, scene=frame)
        else:
            image = frame

        # Display the potentially-modified frame.
        cv2.imshow("frame", image) 
        
        # Press the "q" key to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break


if __name__ == "__main__":
    # Load segmentation model
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl", detection_speed = "rapid")

    # Target classes to be erased.
    target_classes = ins.select_target_classes(person=True)

    background = get_background()

    video_stream(model=ins, ignored_classes=target_classes, background=background, camera=0)
