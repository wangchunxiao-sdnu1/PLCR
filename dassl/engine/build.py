from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")  # 创建注册表
# 后面还有一句
'''
To register an object: 要注册对象：

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone(nn.Module):
这句话在coop.py中出现，如下：

@TRAINER_REGISTRY.register()  
class CoOp(TrainerX):

也就是说trainer=coop的注册器是在coop.py那个类中？
'''


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names() # 可用的trainers是注册表中出现的trainer的名字
    check_availability(cfg.TRAINER.NAME, avai_trainers)  # 检查列表中是否有可用的名字，trainer.name=coop，判断它是否可用，
    # 这里可以通过coop的注册表过去coop.py那边
    if cfg.VERBOSE:  # 日志显示，=1，以进度条形式显示
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))  # 进行格式化替换，用后边这个.format()的结果去替换前边{}中的结果。cfg.TRAINER.NAME=CoOp，
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)   # 是从这里得到

