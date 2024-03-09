from configs import config
#rom trainer.train_baseline import main
from trainer.train_att import main
args = config()
main(args)