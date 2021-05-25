from model import ObjectDetector
from helpfunctions import creating_rectangles
from helpfunctions import IOU
from helpfunctions import distance
import torch.optim as optim
from torch import nn
import torch
import torchvision
import numpy as np
from dataloader import Dataset
import matplotlib.pyplot as plt
import matplotlib

img_size=8
net = ObjectDetector()
num_objects=1
num_imgs=50000
optimizer = optim.Adadelta(net.parameters(),lr=0.01)
criterion = nn.MSELoss()
batch_size=num_imgs
testset = Dataset("test",1)
trainloader = torch.utils.data.DataLoader(Dataset("train",1),batch_size=32,shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)
train=True
if train:
    i = int(0.8 * num_imgs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if train:
        for epoch in range (50):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.float().backward()
                optimizer.step()
                running_loss += loss.item()

        torch.save(net.state_dict(), "test.pth")
net = ObjectDetector()
net.load_state_dict(torch.load("test.pth"))
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
test_bboxes,test_imgs=testset.get_imgs_bbox()
with torch.no_grad():
   pred_bboxes = np.asarray([net(x) for x,y in testloader]) * 8
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)

plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0][0][0], pred_bbox[0][0][1]), pred_bbox[0][0][2], pred_bbox[0][0][3], ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox[0][0], exp_bbox)), (pred_bbox[0][0][0], pred_bbox[0][0][1]+pred_bbox[0][0][3]+0.2), color='r')
plt.show()