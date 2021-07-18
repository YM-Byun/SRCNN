import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, type='rgb'):
        super(SRCNN, self).__init__()

        C = 3

        if type == 'ycbcr':
            C = 1

        self.patch_extraction_representation = nn.Sequential(
            nn.Conv2d(in_channels=C,
                out_channels=64,
                kernel_size=9),
            nn.ReLU(True))

        self.non_linear_mapping = nn.Sequential(
            nn.Conv2d(in_channels=64,
                out_channels=32,
                kernel_size=1),
            nn.ReLU(True))

        self.reconstruction = nn.Sequential(
            nn.Conv2d(in_channels=32,
                out_channels=C,
                kernel_size=5))

    def forward(self, x):
        x = self.patch_extraction_representation(x)
        x = self.non_linear_mapping(x)
        x = self.reconstruction(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(10, 1, 32, 32)
  
    srcnn = SRCNN()

    print ("SRCNN network")
    print (srcnn)

    print ("\n--------------------------------------\n")

    x = srcnn(dummy_data)

    print (f"Result: {x.shape}")
