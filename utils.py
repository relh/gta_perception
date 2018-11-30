from pathlib import Path
import torch

from tqdm import tqdm


class Runner(object):
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

    def _iteration(self, data_loader, batch_size, is_train=True):
        loop_loss = []
        accuracy = []
        outputs = []
        pbar = tqdm(data_loader, ncols=40, disable=False)
        for i, (path, data, target) in enumerate(pbar):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)

            # Testing is with batch_size 1
            if not is_train:
                for p in range(len(path)):
                  outputs.append((path[p], int(output.data.max(1)[1][p])))

            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Fetch LR
            lr = 0.0
            for param_group in self.optimizer.param_groups:
              lr = param_group['lr']

            # Set Progress bar
            pbar.set_description(
                "{} epoch {}: itr {:<5}/ {} - loss {:.3f} - accuracy {:.2f}% - lr {:.4f}"
                .format('TRAIN' if is_train else 'TEST', self.epoch, i*batch_size, len(data_loader)*batch_size, loss.data.item(), (sum(accuracy) / ((i+1)*batch_size))*100.0, lr))

        mode = "train" if is_train else "test/val"
        if mode == "test/val":
          with open('test_track.csv', 'a') as f:
            f.write(f">>>[{mode}] epoch: {self.epoch} loss: {sum(loop_loss):.2f}/accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}\n")
        if is_train:
          return loop_loss, accuracy, None
        else:
          return loop_loss, accuracy, outputs

    def train(self, data_loader, batch_size):
        self.model.train()
        with torch.enable_grad():
            loss, correct, _ = self._iteration(data_loader, batch_size)

    def test(self, data_loader, batch_size):
        self.model.eval()
        with torch.no_grad():
            loss, correct, outputs = self._iteration(data_loader, batch_size, is_train=False)
        return outputs, loss

    def loop(self, epochs, train_data, test_data, scheduler, batch_size):
        for ep in range(1, epochs + 1):
            self.epoch = ep
            self.train(train_data, batch_size)
            _, loss = self.test(test_data, batch_size)
            if scheduler is not None:
                scheduler.step(sum(loss))
            if ep % self.save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch, "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
