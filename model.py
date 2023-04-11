import torch
import torch.nn as nn
#from torchsummary import summary
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import timm
from facenet_pytorch import InceptionResnetV1

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.net = timm.create_model('resnet50', pretrained=pretrained)
        # the output size of this model
        self.out_features = self.net.fc.in_features
    def forward(self, x):
        return self.net.forward_features(x)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class FaceNet(nn.Module):
    def __init__(self, num_classes, model_name=None, pool=None, dropout=0.0, embedding_size=512, device='cuda', pretrain=True):
        super(FaceNet, self).__init__()
        # Backbone (backbone)
        # three models choice 1. SE-ResNeXt101 2.EfficientNetB7 3.InceptionResnetV1 (Pre-trained for face recog.)
        self.model_name = model_name

        # model
        if model_name == "InceptionResnetV1":
            if pretrain == False:
                self.model = InceptionResnetV1(pretrain)
            else:
                self.model = InceptionResnetV1(pretrained='vggface2')
        else:
            self.model = ResNet50(pretrain)

        # global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # neck
        self.neck = nn.Sequential(
                nn.Linear(self.model.out_features, embedding_size, bias=True),
                nn.BatchNorm1d(embedding_size, eps=0.001),
            )
        self.dropout = nn.Dropout(p=dropout)

        self.head = ArcMarginProduct(embedding_size, num_classes)
        
    def forward(self, x):   
        # backbone
        if self.model_name == None:
            embeddings = self.model(x)
            logits = self.head(embeddings)
            return {'logits': logits, 'embeddings': embeddings}

        x = self.model(x)
        # global pool
        x = self.global_pool(x)
        x = self.dropout(x)
        # change the output from cnn to a vector first
        x = x[:,:,0,0]
        # neck
        embeddings = self.neck(x)
        # vector with num_classes
        logits = self.head(embeddings)
        return {'logits': logits, 'embeddings': embeddings}