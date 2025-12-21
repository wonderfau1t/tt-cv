import os

import wget


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


if __name__ == "__main__":
    videos_dir = "videos"

    common_url = "https://lab.osai.ai/datasets/openttgames/data/"

    test_video_filenames = ["test_{}.mp4".format(i) for i in range(1, 7)]  # 1 - 7

    for video_fn in test_video_filenames:
        if not os.path.isfile(os.path.join(videos_dir, video_fn)):
            print("Downloading...{}".format(common_url + video_fn))
            wget.download(common_url + video_fn, os.path.join(videos_dir, video_fn))
