# run in AWS SageMaker 
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

# 获取执行角色
role = sagemaker.get_execution_role()

# S3 路径配置
bucket = 'floorplan-dataset.wm'
prefix = 'yolov11-allobjectdetect'

s3_data = f's3://{bucket}/Floor Plan All Objects.v1i.yolov11/'
s3_pretrained = f's3://{bucket}/models/'
s3_output = f's3://{bucket}/outputs/{prefix}/'

# 输入通道配置
inputs = {
    'train': TrainingInput(s3_data),
    'pretrained': TrainingInput(s3_pretrained)
}

# 创建训练任务
estimator = PyTorch(
    entry_point='train.py',
    source_dir='code',  # 包含 train.py 和 requirements.txt
    role=role,
    framework_version="2.0.1",
    py_version="py310",
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    hyperparameters={},
    dependencies=['code/requirements.txt'],  # 指定依赖文件
    output_path=s3_output,
    base_job_name='yolov11-allobjectdetect',
)

# 启动训练
estimator.fit(inputs)
