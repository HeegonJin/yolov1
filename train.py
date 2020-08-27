from dataset import VOC
from torchvision import transforms
import torch
import os
import numpy as np
import yolov1
from yolov1 import detection_loss_4_yolo

def detection_collate(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])

        np_label = np.zeros((7, 7, 6), dtype=np.float32)
        for object in sample[1]:
            objectness = 1
            classes = object[0]
            x_ratio = object[1]
            y_ratio = object[2]
            w_ratio = object[3]
            h_ratio = object[4]

            scale_factor = (1 / 7)
            grid_x_index = int(x_ratio // scale_factor)
            grid_y_index = int(y_ratio // scale_factor)
            x_offset = (x_ratio / scale_factor) - grid_x_index
            y_offset = (y_ratio / scale_factor) - grid_y_index

            np_label[grid_x_index][grid_y_index] = np.array([objectness, x_offset, y_offset, w_ratio, h_ratio, classes])

        label = torch.from_numpy(np_label)
        targets.append(label)
    return torch.stack(imgs, 0), torch.stack(targets, 0)


data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "VOCdevkit", "VOC2007")
train_dataset = VOC(root=data_path,
                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True,
                                           collate_fn=detection_collate)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = yolov1.YOLOv1(params={"dropout": 0.5, "num_class": 20})
if device.type == 'cpu':
    model = torch.nn.DataParallel(net)
else:
    model = torch.nn.DataParallel(net, device_ids=0).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

num_epochs = 1000
total_step = len(train_loader)
total_train_step = num_epochs * total_step

for epoch in range(num_epochs):

    if (epoch % 200) == 0:
        scheduler.step()

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calc Loss
        loss, \
        obj_coord1_loss, \
        obj_size1_loss, \
        obj_class_loss, \
        noobjness1_loss, \
        objness1_loss = detection_loss_4_yolo(outputs, labels, device.type)
        # objness1_loss = detection_loss_4_yolo(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())