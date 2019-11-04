
from PIL import Image
from mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
from mtcnn_pytorch.src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn_pytorch.src.first_stage import run_first_stage
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from lz import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = 'cpu'

class MTCNN():
    def __init__(self):
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square=True)
    
    def share_memory(self):
        self.pnet.share_memory()
        self.rnet.share_memory()
        self.onet.share_memory()
    
    def align(self, img):
        _, landmarks = self.detect_faces(img)
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112, 112))
        # cvb.show_img(warped_face)
        return Image.fromarray(warped_face)
    
    def align_best(self, img, limit=None, min_face_size=20., **kwargs):
        try:
            boxes, landmarks = self.detect_faces(img, min_face_size,)
            img = to_numpy(img)
            if limit:
                boxes = boxes[:limit]
                landmarks = landmarks[:limit]
            nrof_faces = len(boxes)
            boxes = np.asarray(boxes)
            if nrof_faces > 0:
                det = boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                bindex = 0
                if nrof_faces > 1:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    bindex = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                boxes = boxes[bindex, 0:4]
                landmarks = landmarks[bindex, :]
                facial5points = [[landmarks[j], landmarks[j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112, 112))
                return to_image(warped_face)
            else:
                logging.warning(f'no face detected, {kwargs} ')
                return to_image(img).resize((112, 112), Image.BILINEAR)
        except Exception as e:
            logging.warning(f'face detect fail, err {e}')
            return to_image(img).resize((112, 112), Image.BILINEAR)
    
    def detect_faces(self, image, min_face_size=20.,
                     # thresholds=[0.7, 0.7, 0.8],
                     thresholds=[0.1, 0.1, 0.9],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """
        image = to_image(image)
        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)
        
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        
        # scales for scaling the image
        scales = []
        
        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m
        
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1
        
        # STAGE 1
        
        # it will be returned
        bounding_boxes = []
        
        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)
            
            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)
            
            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]
            
            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]
            
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            
            # STAGE 2
            
            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            
            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]
            thresh = thresholds[1]
            keep = np.where(probs[:, 1] > thresh)[0]
            # while keep.shape[0] == 0:
            #     thresh -= 0.01
            #     keep = np.where(probs[:, 1] > thresh)[0]
            # print('2 stage thresh', thresh)
            
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            
            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            
            # STAGE 3
            
            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
            
            thresh = thresholds[2]
            keep = np.where(probs[:, 1] > thresh)[0]
            if len(keep) == 0:
                return [], []
            # while keep.shape[0] == 0:
            #     thresh -= 0.01
            #     keep = np.where(probs[:, 1] > thresh)[0]
            # print('3 stage one thresh', thresh)
            
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]
            
            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
            
            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]
        
        return bounding_boxes, landmarks
