from utils.config import parse_args
from utils.data_loader import get_data_loader
from models.nk_model import nkModel, Dvector
import torch
from utils.engine import train, val
from torch.utils.tensorboard import SummaryWriter
# main.py
# config.py
def main(args):
    summary = SummaryWriter(args.summary_location)
    train_loader, val_loader = get_data_loader(args)
    # test_loader 만들어야 한다
    model = nkModel(args)
    #model = Dvector(args)
    # if model cuda
    model = model.cuda()
    # if model parallel
    # model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.load_model:
        location = args.load_model_location
        print("load model : ", location)
        checkpoint = torch.load(location)
        # # load params
        model.load_state_dict(checkpoint['model_state_dict'])


    for epoch in range(args.start_epoch, args.max_epoch):
        train(model, train_loader, optimizer, epoch, summary)
        val(model, val_loader, epoch, summary)
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },  '/data2/sound_model/selectTF_model_{}.pth'.format(epoch))

if __name__ == '__main__':
    config = parse_args()
    main(config)

