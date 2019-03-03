import numpy as np
import matplotlib.pyplot as plt

T = 22709 * 9
# T = 200 * 391
T = int(T)
t = np.arange(1, T)
gamma = 1e-3
low = (1 - 1 / (gamma * t + 1))
high = (1 + 1 / (gamma * t))
plt.plot(t, low)
plt.plot(t, high)
plt.yscale('log')
# plt.show()

# for ind in [T//8, T*3//4, ]:
for ind in [T * 2 // 9, T * 5 // 9, T * 7 // 9]:
    print(high[ind] - 1)

'''
class TestData(torch.utils.data.Dataset):
    def __init__(self):
        self.imgfn_iter = itertools.chain(
            glob.glob(args.data_dir + '/**/*.jpg', recursive=True),
            glob.glob(args.data_dir + '/**/*.JPEG', recursive=True))
    
    def __len__(self):
        return int(10 * 10 ** 6)  # assume ttl test img less than 10M
    
    def __getitem__(self, item=None):
        try:
            imgfn = next(self.imgfn_iter)
            finish = 0
            img = cvb.read_img(imgfn)  # bgr
            # img = cvb.bgr2rgb(img)
            img = cvb.resize(img, (336, 336),
                             interpolation=cv2.INTER_CUBIC)  # assume the img has been processed and extended
        except StopIteration:
            # logging.info(f'folder iter end')
            imgfn = ""
            finish = 1
            img = np.zeros((336, 336, 3), dtype=np.uint8)
        return {'imgfn': imgfn,
                "finish": finish,
                'img': img}


loader = torch.utils.data.DataLoader(TestData(), batch_size=args.batch_size,
                                     num_workers=24,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False
                                     )
for ind, data in enumerate(loader):
    if (data['finish'] == 1).all().item():
        logging.info('finish')
        break
'''