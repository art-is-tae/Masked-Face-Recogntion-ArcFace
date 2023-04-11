from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class customized_dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str, label_to_samples=None):
        self.df = dataframe
        self.mode = mode

        transforms_train = [transforms.Resize((128,128)),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
        
        transforms_test = [transforms.Resize((128,128)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
        
        self.transforms_train = transforms.Compose(transforms_train)
        self.transforms_test = transforms.Compose(transforms_test)
        #self.label_to_samples = np.array(label_to_samples)
        self.label_to_samples = label_to_samples

    def __len__(self) -> int:
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # target label
        target = self.df.iloc[index]['target']
        image_path = self.df.iloc[index]['path']

        # original image
        img = Image.open(image_path).convert('RGB')

        if self.mode=='train': # train
            img = self.transforms_train(img)
            return {'image':img, 'target':target}
        
        else: # test and eval
            img = self.transforms_test(img)
            pair_path = self.df.iloc[index]['pair_path']
            pair_target = self.df.iloc[index]['pair_target']
            pair_img = Image.open(pair_path).convert('RGB')
            pair_img = self.transforms_test(pair_img)
            return {'image':img, 'target':target, 'pair_image':pair_img, 'pair_target':pair_target}
        # else: # test
        #    img = self.transforms_test(img)
        #    return {'image':img, 'target':target}