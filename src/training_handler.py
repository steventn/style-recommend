import timm
import torch
import torchmetrics
import os
from tqdm import tqdm
from src.model_handler import ModelHandler


class TrainingHandler:
    @staticmethod
    def calculate_metrics(model, images, targets, loss_function, epoch_loss, epoch_accuracy, epoch_f1, f1_score):
        predictions = model(images)
        loss = loss_function(predictions, targets)
        return loss, epoch_loss + loss.item(), epoch_accuracy + (
                torch.argmax(predictions, dim=1) == targets).sum().item(), epoch_f1 + f1_score(predictions, targets)

    @staticmethod
    def train_model(classes, train_data_loader, val_data_loader, device, root):
        save_prefix, save_dir = "ecommerce", os.path.join(root, "models")
        model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(classes)).to(device)
        model, epochs, device, loss_function, optimizer = ModelHandler.setup_training(model, device)
        f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)
        print("Start training...")
        best_accuracy, best_loss, threshold, not_improved, patience = 0, float("inf"), 0.01, 0, 5
        train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = [], [], [], [], [], []

        best_loss = float(torch.inf)

        for epoch in range(epochs):
            epoch_loss, epoch_accuracy, epoch_f1 = 0, 0, 0
            for idx, batch in tqdm(enumerate(train_data_loader)):
                images, targets = ModelHandler.move_to_device(batch, device)

                loss, epoch_loss, epoch_accuracy, epoch_f1 = TrainingHandler.calculate_metrics(model, images, targets,
                                                                                               loss_function,
                                                                                               epoch_loss,
                                                                                               epoch_accuracy,
                                                                                               epoch_f1,
                                                                                               f1_score_metric)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_to_track = epoch_loss / len(train_data_loader)
            train_accuracy_to_track = epoch_accuracy / len(train_data_loader.dataset)
            train_f1_to_track = epoch_f1 / len(train_data_loader)
            train_losses.append(train_loss_to_track)
            train_accuracies.append(train_accuracy_to_track)
            train_f1_scores.append(train_f1_to_track)

            print(f"{epoch + 1}-epoch train process is completed!")
            print(f"{epoch + 1}-epoch train loss          -> {train_loss_to_track:.3f}")
            print(f"{epoch + 1}-epoch train accuracy      -> {train_accuracy_to_track:.3f}")
            print(f"{epoch + 1}-epoch train f1-score      -> {train_f1_to_track:.3f}")

            model.eval()
            with torch.no_grad():
                val_epoch_loss, val_epoch_accuracy, val_epoch_f1 = 0, 0, 0
                for idx, batch in enumerate(val_data_loader):
                    images, targets = ModelHandler.move_to_device(batch, device)
                    loss, val_epoch_loss, val_epoch_accuracy, val_epoch_f1 = TrainingHandler.calculate_metrics(model,
                                                                                                               images,
                                                                                                               targets,
                                                                                                               loss_function,
                                                                                                               val_epoch_loss,
                                                                                                               val_epoch_accuracy,
                                                                                                               val_epoch_f1,
                                                                                                               f1_score_metric)

                val_loss_to_track = val_epoch_loss / len(val_data_loader)
                val_accuracy_to_track = val_epoch_accuracy / len(val_data_loader.dataset)
                val_f1_to_track = val_epoch_f1 / len(val_data_loader)
                val_losses.append(val_loss_to_track)
                val_accuracies.append(val_accuracy_to_track)
                val_f1_scores.append(val_f1_to_track)

                print(f"{epoch + 1}-epoch validation process is completed!")
                print(f"{epoch + 1}-epoch validation loss     -> {val_loss_to_track:.3f}")
                print(f"{epoch + 1}-epoch validation accuracy -> {val_accuracy_to_track:.3f}")
                print(f"{epoch + 1}-epoch validation f1-score -> {val_f1_to_track:.3f}")

                if val_loss_to_track < (best_loss + threshold):
                    os.makedirs(save_dir, exist_ok=True)
                    best_loss = val_loss_to_track
                    torch.save(model.state_dict(), f"{save_dir}/{save_prefix}_best_model_test_sub.pth")
                else:
                    not_improved += 1
                    print(f"Loss value did not decrease for {not_improved} epochs")
                    if not_improved == patience:
                        print(f"Stop training since loss value did not decrease for {patience} epochs.")
                        break
