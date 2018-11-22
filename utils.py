from pathlib import Path
import torch

from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=5):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.epoch = 0

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        outputs = []
        for i, (path, data, target) in enumerate(tqdm(data_loader, ncols=160, disable=True)):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            outputs.append((path, output.data.max(1)[1]))
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % 10 == 0:
              print("{} epoch {}: \t itr {:<5}/ {} \t loss {:.2f} \t accuracy {:.3f} \t it/s {:.2f} \t lr {:.3f}"\
                  .format('TRAIN' if is_train else 'TEST', self.epoch, i, len(data_loader), loss.data.item(), sum(accuracy) / (i+1), 1.0, 1.0))

        mode = "train" if is_train else "test"
        print(f">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}")
        if is_train:
          return loop_loss, accuracy, None
        else:
          return loop_loss, accuracy, outputs

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            loss, correct, _ = self._iteration(data_loader)

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss, correct, outputs = self._iteration(data_loader, is_train=False)
        return outputs

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.train(train_data)
            self.test(test_data)
            if ep % self.save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
