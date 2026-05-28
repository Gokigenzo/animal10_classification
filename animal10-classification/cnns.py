import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Cập nhật thông số chuẩn AlexNet cho ảnh kích thước 224x224
        self.conv1 = self.make_block(in_c_Conv=3, out_c_Conv=96, ker_size_Conv=11, stride_Conv=4, padding_Conv=2, ker_size_pool=3, stride_pool=2)
        self.conv2 = self.make_block(in_c_Conv=96, out_c_Conv=256, ker_size_Conv=5, stride_Conv=1, padding_Conv=2, ker_size_pool=3, stride_pool=2) 
        self.conv3 = self.make_block(in_c_Conv=256, out_c_Conv=384, ker_size_Conv=3, stride_Conv=1, padding_Conv=1, ker_size_pool=None, stride_pool=None)
        self.conv4 = self.make_block(in_c_Conv=384, out_c_Conv=384, ker_size_Conv=3, stride_Conv=1, padding_Conv=1, ker_size_pool=None, stride_pool=None)
        self.conv5 = self.make_block(in_c_Conv=384, out_c_Conv=256, ker_size_Conv=3, stride_Conv=1, padding_Conv=1, ker_size_pool=3, stride_pool=2)

        # Sử dụng AdaptiveAvgPool2d giúp cố định đầu ra luôn là (6, 6) 
        # Tránh việc lỗi kích thước nếu sau này bạn đổi sang ảnh 225x225 hoặc kích thước gần bằng
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), # Thêm dropout để tránh quá khớp (overfitting)
            nn.Linear(in_features=256 * 6 * 6, out_features=4096), # 256 * 6 * 6 = 9216
            nn.LeakyReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.LeakyReLU(),

            # Loại bỏ nn.ReLU() ở tầng cuối cùng để sửa lỗi mất gradient (Lỗi đứng im loss đã phân tích trước đó)
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    
    def make_block(self, in_c_Conv, out_c_Conv, ker_size_Conv, stride_Conv, padding_Conv=0, ker_size_pool=None, stride_pool=None):
        if ker_size_pool is not None or stride_pool is not None:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_c_Conv, out_channels=out_c_Conv, kernel_size=ker_size_Conv, stride=stride_Conv, padding=padding_Conv),
                nn.BatchNorm2d(num_features=out_c_Conv),    
                nn.MaxPool2d(kernel_size=ker_size_pool, stride=stride_pool),
                nn.LeakyReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_c_Conv, out_channels=out_c_Conv, kernel_size=ker_size_Conv, stride=stride_Conv, padding=padding_Conv),
                nn.BatchNorm2d(num_features=out_c_Conv),
                nn.LeakyReLU() 
            ) 
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Sử dụng flatten an toàn và chuyên nghiệp hơn x.view()
        x = self.fc(x)
        
        return x
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c,  kernel_size=3, padding = "same"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

        )
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels= out_c, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_c)
            )
        else: 
            self.shortcut = nn.Identity()

        
    def forward(self, x):
        output = self.block(x)
        x = self.shortcut(x)
        output = x + output
        output = nn.ReLU()(output)
        return output


class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 64, kernel_size=7, stride= 2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2) 
        )
        self.layer1 = self.make_block(2, in_c=64, out_c=64, stride=1)
        self.layer2 = self.make_block(2, in_c=64, out_c=128, stride=2)
        self.layer3 = self.make_block(2, in_c=128, out_c=256, stride=2)
        self.layer4 = self.make_block(2, in_c=256, out_c=512, stride=2)
        self.adt = nn.AdaptiveAvgPool2d((7,7))
        self.fc = nn.Linear(in_features= 7*7*512, out_features=num_classes)

    def make_block(self, num_block, in_c, out_c, stride):
        blocks = [BasicBlock(in_c=in_c,out_c= out_c, stride = stride)]
        for _ in range(num_block-1):
            blocks.append(BasicBlock(in_c=out_c, out_c= out_c, stride=1))
        return nn.Sequential(*blocks)

        

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adt(x)
        x= torch.flatten(x,1)
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    data = torch.rand((1,3,224,224))
    model = ResNet()
    model.forward(data)
    