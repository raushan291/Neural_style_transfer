import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 256 if torch.cuda.is_available() else 128

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_style_features = ['0', '5', '10', '19', '28']    # Initial layer features contains style of img
        self.chosen_content_features = ['21']                        # Deep layer features contains content features of img
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x, content_part=False):
        style_features = []
        content_features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)            # pass x as input to NN layer and store output return by that layer to variable x.

            if content_part and str(layer_num) in self.chosen_content_features:
                content_features.append(x)
                return content_features

            if str(layer_num) in self.chosen_style_features:
                style_features.append(x)
        
        return style_features

    
def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

model = VGG().to(device=device).eval()

# input data
content_image = load_image('/home/rakumar/Neural_style_transfer/inputs/content.jpeg')
style_image = load_image('/home/rakumar/Neural_style_transfer/inputs/style.jpg')
# generated_image = torch.rand(content_image.shape, device=device, requires_grad=True)   # random noise
generated_image = content_image.clone().requires_grad_(True)

# hyperparameters
total_epochs = 2
learning_rate = 0.001
alpha = 1
beta = 0.001
optimizer = optim.Adam([generated_image], lr=learning_rate)

# Generate images...
for epoch in range(total_epochs):
    content_image_contents = model(content_image, content_part=True)              # length = 1
    generated_image_contents = model(generated_image, content_part=True)          # length = 1

    style_image_style = model(style_image)                                        # length = 5
    generated_image_style = model(generated_image)                                # length = 5

    style_loss = content_loss = 0
    
    # content loss
    for cont_img_content, gen_img_content in zip(content_image_contents, generated_image_contents):
        content_loss += torch.mean( ( gen_img_content - cont_img_content) ** 2)

    # style part of images..
    for styleImg_feature, genImg_feature in zip(style_image_style, generated_image_style):
        
        batch_size, channel, height, width = genImg_feature.shape

        # Gram Marix (for style loss calculation..)
        G = genImg_feature.view(channel, height*width).mm( genImg_feature.view(channel, height*width).t() )
        S = styleImg_feature.view(channel, height*width).mm( styleImg_feature.view(channel, height*width).t() )

        # style loss
        style_loss += torch.mean((G - S) **2 )
    
    total_loss = alpha*content_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == total_epochs-1:
         print(total_loss)
         save_image(generated_image, 'generated_'+str(epoch)+'.jpg')
