import lz, argparse
import torch, glob, cvbase as cvb
from torchvision import transforms as trans
from mtcnn import MTCNN
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data2/share/fakeimgnet/train')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

# print(len(glob.glob(args.data_dir + '/**', recursive=True)))
# class DatasetOPPO(torch.utils.data.Dataset):
#     def __init__(self):
#         self.test_transform = trans.Compose([
#             trans.ToTensor(),
#             trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
mtcnn = MTCNN()

for imgfn in glob.glob(args.data_dir + '/**', recursive=True):
    img = cvb.read_img(imgfn) # bgr
    img = cvb.bgr2rgb()
    