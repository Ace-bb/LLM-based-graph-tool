
#   pip install openxlab #安装

#   pip install -U openxlab #版本升级

# from ..conf.settings import OPENXLAB_AK, OPENXLAB_SK
import openxlab
openxlab.login(ak="r46ko7b4xanaz44yolb3", sk="y8lojrapzpvqwgynykbybydkkxdzqr3xeke40nlm") #进行登录，输入对应的AK/SK

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/TableBank') #数据集信息及文件列表查看

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/TableBank', target_path='/root/LLM-based-graph-tool/data/datasets/')  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/TableBank',source_path='/README.md', target_path='/root/LLM-based-graph-tool/data/datasets/') #数据集文件下载