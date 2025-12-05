## 

## Congifuration Environment
- có thể dùng trình conda để quản lý môi trường và dùng wsl
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Pre-training

**Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is utilized for face detection and face alignment. All the detected faces are normlaize to 256$\times$256$\times$3, where only RGB channels are utilized for training. 

To be specific, we process every frame of each video and then utilize the `sample_frames` function in the `utils/utils.py` to sample frames during training.

Put the processed frames in the path `$root/data/dataset_name`.

**bước này đã làm xong và lưu ở data_label**
**Data Label Generation.** 

Move to the `$root/data_label` and generate the data label list:
```python
python generate_label.py
```

## Training

Move to the folder `$root/experiment/testing_scenarios/` and just run like this:
```python
python train_ssdg_full.py
```

The file `config.py` contains all the hype-parameters used during training.

## Testing

Run like this:
```python
python dg_test.py
```

# các việc cần làm bây giờ 

## Lý do:
Ta chỉ có 3/4 bộ datasets nên ta cần modify lại các hàm ở bước training và testing sao cho thích ứng với 3 bộ dataset đó

## Các bước làm 
### Download 3 bộ datasets đó thông qua drive

#### Cách 1: Tải về Server Linux / Colab (Khuyên dùng cho AI)
Dùng wget hoặc curl để tải link Google Drive thường hay bị lỗi (do cơ chế xác thực virus scan với file lớn). Công cụ chuẩn nhất là gdown.

Người dùng kia cần làm:

##### 1. Cài đặt gdown Chạy lệnh sau trên server để cài công cụ (yêu cầu server đã có Python/Pip):

```Bash
pip install gdown
```
##### 2. Chạy lệnh tải folder Sử dụng lệnh sau để tải toàn bộ folder về. Lưu ý tham số --folder là bắt buộc vì đây là tải thư mục.

```Bash
gdown https://drive.google.com/drive/folders/1y1RCZfVSbCRSsc7r5BYaQb4-V4GTofvT -O ./Ten_Folder_Muon_Luu --folder
```

-O ./Ten_Folder_Muon_Luu: Đặt tên cho thư mục sẽ lưu trên server (ví dụ: dataset_CVPR).

--folder: Báo cho gdown biết đây là link folder chứ không phải file lẻ.
#### Cách 2: Tải về máy tính cá nhân (Windows/Mac)
Tải data từ đường link: https://drive.google.com/drive/folders/1y1RCZfVSbCRSsc7r5BYaQb4-V4GTofvT

Nhấn nút Download trên trình duyệt.




