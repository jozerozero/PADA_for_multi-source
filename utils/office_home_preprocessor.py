import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

from utils.prepare_dataset import check_exits, download_data


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for domain adaptation in image classification

        Parameters:
            - **root** (str): Root directory of dataset
            - **classes** (List[str]): The names of all the classes
            - **data_list_file** (str): File to read the image list from.
            - **transform** (callable, optional): A function/transform that  takes in an PIL image \
                and returns a transformed version. E.g, ``transforms.RandomCrop``.
            - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

        .. note:: In `data_list_file`, each line 2 values in the following format.
            ::
                source_dir/dog_xxx.png 0
                source_dir/cat_123.png 1
                target_dir/dog_xxy.png 0
                target_dir/cat_nsdf3.png 1

            The first value is the relative path of an image, and the second value is the label of the corresponding image.
            If your data_list_file has different formats, please over-ride `parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable]=None, target_transform: Optional[Callable]=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = self.parse_data_file(file_name=data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        self.loader = default_loader

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        :param index: (int): Index
        :return: (tuple): (image, target) where target is index of the target class
        """
        path, target = self.data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """
        parse file to data list
        :param file_name: (str): The path of data file
        :return: (list): List of (image path, class_index) tuples
        """

        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """return the number of the classes"""
        return len(self.classes)


class OfficeHome(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

        Parameters:
            - **root** (str): Root directory of dataset
            - **task** (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
                ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
            - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
                in root directory. If dataset is already downloaded, it is not downloaded again.
            - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
                transformed version. E.g, ``transforms.RandomCrop``.
            - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

        .. note:: In `root`, there will exist following files after downloading.
            ::
                Art/
                    Alarm_Clock/*.jpg
                    ...
                Clipart/
                Product/
                Real_World/
                image_list/
                    Art.txt
                    Clipart.txt
                    Product.txt
                    Real_World.txt
        """

    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ee615d5ad5e146278a80/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/81a4f30c7e894298b435/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/d4ad15137c734917aa5c/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/a6b643999c574184bbcd/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/60ca8452bcf743408245/?dl=1")
    ]
    image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    def __init__(self, root: str, task: str, download: Optional[bool]=False,
                 transform: Optional[Callable]=None, label_transform: Optional[Callable]=None):
        assert task in self.image_list

        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root=root, file_name=file_name), self.download_list))

        super(OfficeHome, self).__init__(root=root, classes=self.CLASSES, data_list_file=data_list_file,
                                         transform=transform, target_transform=label_transform)
