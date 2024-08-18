import torch


def get_optimizer(conf_optim):
    optimizer_cls = getattr(
        torch.optim, conf_optim.optimizer.name
    )

    if hasattr(conf_optim, 'lr_scheduler'):
        scheduler_cls = getattr(
            torch.optim.lr_scheduler,
            conf_optim.lr_scheduler.name
        )
    else:
        scheduler_cls = None
        
    return optimizer_cls, scheduler_cls
