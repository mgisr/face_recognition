from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people


def load_data(min_faces_per_person=70, train_split: float = 0.8) -> tuple:
    """加载LFW数据集

    Args:
        min_faces_per_person (int, optional): 指定返回的数据集中每类最小的图片数。默认值为70。
        split (float): 指定数据集中训练集占的比例，为None时不做切分。默认为0.8。
    Returns:
        train_split is None:
            tuple: (data, label) data的格式为(number_of_people, 2914)，其中2914=62*47，是由数据集中的图片展开得到的一维向量。
            label为一维向量，对应于每个data的标签。
        train_split not is None:
            tuple: (train_data, train_label, test_data, test_label)，data，label格式同上。
    """

    data, label = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person, return_X_y=True
    )

    if train_split != None:
        train_data, test_data = train_test_split(data, train_size=train_split)
        train_label, test_label = train_test_split(label, train_size=train_split)
        return (train_data, train_label, test_data, test_label)
    else:
        return (data, label)


def main():
    pass


if __name__ == "__main__":
    main()
