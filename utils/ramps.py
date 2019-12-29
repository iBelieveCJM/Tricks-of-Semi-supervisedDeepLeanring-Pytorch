import numpy as np

def pseudo_rampup(T1, T2):
    def warpper(epoch):
        if epoch > T1:
            alpha = (epoch-T1) / (T2-T1)
            if epoch > T2:
                alpha = 1.0
        else:
            alpha = 0.0
        return alpha
    return warpper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""
    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0
    return warpper


def exp_rampdown(rampdown_length, num_epochs):
    """Exponential rampdown from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5* (epoch - (num_epochs - rampdown_length))
            return float(np.exp(-(ep * ep) / rampdown_length))
        else:
            return 1.0
    return warpper


def cosine_rampdown(rampdown_length, num_epochs):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5* (epoch - (num_epochs - rampdown_length))
            return float(.5 * (np.cos(np.pi * ep / rampdown_length) + 1))
        else:
            return 1.0
    return warpper


def exp_warmup(rampup_length, rampdown_length, num_epochs):
    rampup = exp_rampup(rampup_length)
    rampdown = exp_rampdown(rampdown_length, num_epochs)
    def warpper(epoch):
        return rampup(epoch)*rampdown(epoch)
    return warpper


def test_warmup():
    warmup = exp_warmup(80, 50, 500)
    for ep in range(500):
        print(warmup(ep))
