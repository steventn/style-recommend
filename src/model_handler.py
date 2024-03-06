import torch


class ModelHandler:
    @staticmethod
    def setup_training(model, device):
        return model.to(device).eval(), 5, device, torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=model.parameters(),
                                                                                                 lr=3e-4)

    @staticmethod
    def move_to_device(batch, device):
        return batch[0].to(device), batch[1].to(device)
