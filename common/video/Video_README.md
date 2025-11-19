# dataPipeline_ops/common/video
### 存放关于video的处理算子
1. ffmpeg_cut_op -  ffmpeg 进行视频切片
   - input:file_path, out_path, start, end or duration
2. video_info_op - 视频信息提取
   - input:file_path;
   - output:fps, frame_count, total_duration
3. video_probe_op - 使用 ffprobe 获取视频元数据，如时长和帧率。
   - input: file_path
   - output: total_duration, fps
4. scene_segmenter_op - 根据设定的固定时长，将视频切分为多个片段。
   - input: file_path, total_duration, fps
   - output: segments (包含各切片路径和时间的列表)
5. decord_reader_op - 使用 decord 库读取视频文件，生成视频读取器对象(VideoReader)并获取基础尺寸信息，为后续抽帧操作做准备。
   - input: file_path
   - output: vr (decord VideoReader对象), total_frame, height, width