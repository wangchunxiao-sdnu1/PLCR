import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown

from dassl.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, session="", label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath) # wang

        self._session = session
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def session(self):
        return self._session

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x_E=None, train_x_V =None, train_u=None, val_E=None, val_V=None, test_E=None, test_V=None):
        self._train_x_E = train_x_E  # labeled training data
        self._train_x_V = train_x_V
        self._train_u = train_u  # unlabeled training data (optional)
        self._val_E = val_E  # validation data (optional)
        self._val_V = val_V
        self._test_E = test_E  # test data
        self._test_V = test_V
        self._num_classes_E = self.get_num_classes(train_x_E)
        self._num_classes_V = self.get_num_classes(train_x_V)
        self._lab2cname_E, self._classnames_E = self.get_lab2cname(train_x_E)
        self._lab2cname_V, self._classnames_V = self.get_lab2cname(train_x_V)

    @property
    def train_x_E(self):
        return self._train_x_E

    @property
    def train_x_V(self):
        return self._train_x_V

    @property
    def train_u(self):
        return self._train_u

    @property
    def val_E(self):
        return self._val_E

    @property
    def val_V(self):
        return self._val_V

    @property
    def test_E(self):
        return self._test_E

    @property
    def test_V(self):
        return self._test_V

    @property
    def lab2cname_E(self):
        return self._lab2cname_E

    @property
    def lab2cname_V(self):
        return self._lab2cname_V

    @property
    def classnames_E(self):
        return self._classnames_E

    @property
    def classnames_V(self):
        return self._classnames_V

    @property
    def num_classes_E(self):
        return self._num_classes_E

    @property
    def num_classes_V(self):
        return self._num_classes_V

    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        assert len(source_domains) > 0, "source_domains (list) is empty"
        assert len(target_domains) > 0, "target_domains (list) is empty"
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output
