# dataPipeline_ops/common/image
### 存放关于image的处理算子
1.  FaceDetectOp - 封装 RetinaFace 进行人脸检测，支持针对局部裁剪图(crop)或全图(frame)进行检测。
    -   input: crop (BGR ndarray), frame (RGB ndarray)
    -   output: faces_in_crop (dict), faces_in_frame (dict)

2.  PersonDetectOp - 封装 mmdet 模型进行人体检测，支持 RGB 或 BGR 输入，具备通道自动纠错重试机制，输出规范化的坐标框列表。
    -   input: frame (RGB ndarray) 或 image (BGR ndarray)
    -   output: person_boxes (list of [x1, y1, x2, y2])

3.  PlaceholdersOp - 根据参考图列表和目标图路径，构建适用于 Qwen-VL 等模型的 `placeholders` 结构，包含图片路径及对应的 image_id。
    -   input: input_images (或 input_images_list), output_image
    -   output: placeholders (list of dict), input_images_list, orig_img_path

4.  RetinaFaceOp - 专门针对单张裁剪图(crop)的 RetinaFace 检测封装，支持 BGR/RGB 格式自动处理及灰度图转换。
    -   input: crop_bgr (ndarray) 或 crop_rgb (ndarray)
    -   output: faces_in_crop (dict)

5.  SaveImageOp - 通用图片保存算子，支持将多个 numpy 数组或 PIL Image 对象保存到指定磁盘路径。
    -   input: save_paths (字典结构: `{name: (file_path, image_array)}`)
    -   output: saved_paths (list of names)