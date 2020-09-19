import os
import pandas as pd
import numpy as np
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class FeatureList(datasets.VisionDataset):

    def __init__(self, root: str, classes: List[str], remove_list, root_list):
        super().__init__(root=root, transform=None, target_transform=None)

        self.data = list()
        self.label_list = list()

        for r in root_list:
            tmp_data, tmp_label_list = self.parser_data_file(filepath=os.path.join(root, r), remove_list=remove_list)
            self.data.extend(tmp_data)
            self.label_list.extend(tmp_label_list)

        # self.data, self.label_list = self.parser_data_file(filepath=root,
        #                                                    remove_list=remove_list)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

        self.loader = default_loader

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        :param index: (int): Index
        :return: (tuple): (image, target) where target is the index of the target class
        """
        feature, target = self.data[index]
        return feature, target

    def __len__(self) -> int:
        return len(self.data)

    def parser_data_file(self, filepath: str, remove_list):
        """
        :param filepath: (str): The path of data file
        :return: (list): List of (feature, class index)tuple
        """
        data_list = list()
        label_list = list()

        data = pd.read_csv(filepath, header=None).values
        feature = data[:, :-1]
        # label = np.array(data[:, -1], dtype=np.int32)
        for label in data[:, -1]:
            if label not in remove_list:
                label_list.append(label)

        for x, y in zip(feature, data[:, -1]):
            if y not in remove_list:
                data_list.append((x, y))

        return data_list, label_list

    @property
    def num_classes(self) -> int:
        """return the number of the class"""
        return len(self.classes)


class OfficeHomeFeature(FeatureList):

    feature_list = {
        "Art": "Art_Art.csv",
        "Clipart": "Clipart_Clipart.csv",
        "Product": "Product_Product.csv",
        "Realworld": "RealWorld_RealWorld.csv"
    }

    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    def get_min_max(self, root):
        feature_list = list()
        for _, v in self.feature_list.items():
            datapath = os.path.join(root, v)
            feature = pd.read_csv(datapath).values[:, :-1]
            feature_list.append(feature)

        feature_list = np.concatenate(feature_list, axis=0)

        return np.max(feature_list), np.min(feature_list)

    def __init__(self, root: str, task: str, remove_list, is_source):

        assert task in self.feature_list
        if is_source:
            source_domain_list = list(self.feature_list.keys())
            assert isinstance(source_domain_list, list)
            assert task in source_domain_list
            source_domain_list.remove(task)
            root_list = [self.feature_list[domain_name] for domain_name in source_domain_list]
        else:
            root_list = [self.feature_list[task]]

        super(OfficeHomeFeature, self).__init__(root=root, classes=self.CLASSES,
                                                remove_list=remove_list, root_list=root_list)
