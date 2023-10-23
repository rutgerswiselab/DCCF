# Path
DATA_DIR = '../data/'  # 原始数据文件及预处理的数据文件目录
DATASET_DIR = '../dataset/'  # 划分好的数据集目录
MODEL_DIR = '../model/'  # 模型保存路径
LOG_DIR = '../log/'  # 日志输出路径
RESULT_DIR = '../result/'  # 数据集预测结果保存路径
COMMAND_DIR = '../command/'  # run.py所用command文件保存路径
LOG_CSV_DIR = '../log_csv/'  # run.py所用结果csv文件保存路径

LIBREC_DATA_DIR = '../librec/data/'  # librec原始数据文件及预处理的数据文件目录
LIBREC_DATASET_DIR = '../librec/dataset/'  # librec划分好的数据集目录
LIBREC_MODEL_DIR = '../librec/model/'  # librec模型保存路径
LIBREC_LOG_DIR = '../librec/log/'  # librec日志输出路径
LIBREC_RESULT_DIR = '../librec/result/'  # librec数据集预测结果保存路径
LIBREC_COMMAND_DIR = '../librec/command/'  # run_librec.py所用command文件保存路径
LIBREC_LOG_CSV_DIR = '../librec/log_csv/'  # run_librec.py所用结果csv文件保存路径

# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.csv'  # 训练集文件后缀
VALIDATION_SUFFIX = '.validation.csv'  # 验证集文件后缀
TEST_SUFFIX = '.test.csv'  # 测试集文件后缀
INFO_SUFFIX = '.info.json'  # 数据集统计信息文件后缀
USER_SUFFIX = '.user.csv'  # 数据集用户特征文件后缀
ITEM_SUFFIX = '.item.csv'  # 数据集物品特征文件后缀
TRAIN_GROUP_SUFFIX = '.train_group.csv'  # 训练集用户交互按uid合并之后的文件后缀
TRAIN_POS_SUFFIX = '.train_pos.csv'  # 训练集用户交互按uid合并之后的文件后缀
VT_POS_SUFFIX = '.vt_pos.csv'  # 验证集和测试集用户交互按uid合并之后的文件后缀
VT_GROUP_SUFFIX = '.vt_group.csv'  # 验证集和测试集用户交互按uid合并之后的文件后缀
RANK_FILE_NAME = 'rank.csv'     # Trained model generated ranking list

VARIABLE_SUFFIX = '.variable.csv'  # ProLogic 变量文件后缀

C_HISTORY = 'history'  # 历史记录column名称
C_HISTORY_LENGTH = 'history_length'  # 历史记录长度column名称
C_HISTORY_NEG = 'history_neg'  # 负反馈历史记录column名称
C_HISTORY_POS_TAG = 'history_pos_tag'  # 用于记录一个交互列表是正反馈1还是负反馈0
C_CTF_HISTORY = 'ctf_history' # counterfactual history column name
C_CTF_HISTORY_LENGTH = 'ctf_history_length'

C_SENT = 'sent'  # 逻辑表达式column名称

# # DataProcessor/feed_dict
X = 'X'
Y = 'Y'
K_IF_TRAIN = 'train'
K_DROPOUT = 'dropout'
K_SAMPLE_ID = 'sample_id'  # 在训练（验证、测试）集中，给每个样本编号。这是该column在data dict和feed dict中的名字。

# Hash
K_ANCHOR_USER = 'anchor_user'  # hash模型用到的anchor user列名
K_UID_SEG = 'uid_seg'  # hash模型用到的聚合uid，分隔不同uid的列名
K_SAMPLE_HASH_UID = 'sample_hash_pos'  # hash模型用到的sample的uid桶的位置

# ProLogic
K_X_TAG = 'x_tag'  # 逻辑模型用到的，以区分变量是否取非
K_OR_LENGTH = 'or_length'  # 逻辑模型用到的，以显示（析取范式中）每个or所连接的合取式中有多少个变量
K_S_LENGTH = 'sent_length'  # 整个逻辑表达式的长度，包括逻辑符号

# # out dict
PRE_VALUE = 'pre_value'
PREDICTION = 'prediction'  # 输出预测
CHECK = 'check'  # 检查中间结果
LOSS = 'loss'  # 输出损失
LOSS_L2 = 'loss_l2'  # 输出l2损失
EMBEDDING_L2 = 'embedding_l2'  # 当前batch涉及到的embedding的l2
L2_BATCH = 'l2_batch'  # 当前计算的embedding的l2的batch大小
CTF_PREDICTION = 'ctf_prediction'
CTF_HIS_DIST = 'ctf_history_distance'
DIM = 'dim'

TIME = 'time'  # 时间column名称
RANK = 'rank'
REAL_BATCH_SIZE = 'real_batch_size'
TOTAL_BATCH_SIZE = 'total_batch_size'
TRAIN = 'train'
DROPOUT = 'dropout'
SAMPLE_ID = 'sample_id'  # 在训练（验证、测试）集中，给每个样本编号。这是该column在data dict和feed dict中的名字。
IID = 'iid'
UID = 'uid'
PROPENSITY_SUFFIX = '.propensity.npy'
TREAT_EMBEDDING_L2 = 'treat_embedding_l2'
DIFF_EMBEDDING_L2 = 'diff_embedding_l2'
LABEL = 'label'