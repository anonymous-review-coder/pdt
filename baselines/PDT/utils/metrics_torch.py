import torch


def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt((MSE(pred, true)))


def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / true))


def metric_torch(pred, true):
    mae = MAE(pred, true).item()
    mse = MSE(pred, true).item()
    rmse = RMSE(pred, true).item()
    mape = MAPE(pred, true).item()
    mspe = MSPE(pred, true).item()

    return mae, mse, rmse, mape, mspe


class MetricCollector:
    def __init__(self, device='cpu'):
        self.device = device
        self.reset()

    def reset(self):
        self.sum_abs_error = torch.tensor(0.0, device=self.device)
        self.sum_squared_error = torch.tensor(0.0, device=self.device)
        self.sum_abs_per_error = torch.tensor(0.0, device=self.device)
        self.sum_sqr_per_error = torch.tensor(0.0, device=self.device)
        self.total = torch.tensor(0, device=self.device)

    def to(self, device):
        self.device = device
        self.sum_abs_error = self.sum_abs_error.to(device)
        self.sum_squared_error = self.sum_squared_error.to(device)
        self.sum_abs_per_error = self.sum_abs_per_error.to(device)
        self.sum_sqr_per_error = self.sum_sqr_per_error.to(device)
        self.total = self.total.to(device)
        return self

    def update(self, preds, target):
        preds = preds.detach()
        target = target.detach()
        self.sum_abs_error += torch.sum(torch.abs(preds - target))
        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.sum_abs_per_error += torch.sum(torch.abs((preds - target) / target))
        self.sum_sqr_per_error += torch.sum(torch.square((preds - target) / target))
        self.total += target.numel()

    def compute(self):
        mse = self.sum_squared_error / self.total
        return {
            "mae": (self.sum_abs_error / self.total).item(),
            "mse": mse.item(),
            "rmse": torch.sqrt(mse).item(),
            "mape": (self.sum_abs_per_error / self.total).item(),
            "mspe": (self.sum_sqr_per_error / self.total).item(),
        }


def create_metric_collector(device='cpu'):
    collector = MetricCollector(device=device)
    collector.reset()
    return collector
