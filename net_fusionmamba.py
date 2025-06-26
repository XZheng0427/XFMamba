import torch
import torch.nn as nn
import torchvision.models
from torchinfo import summary
from collections import OrderedDict
from models import build_model
from models.fusion_vmamba import Backbone_VSSM, ShallowFusionBlock_v4, CSSFVSSLayer_v5


class ModelWrapper(nn.Module):
    def __init__(self, original_model, output_index=0):
        super(ModelWrapper, self).__init__()
        self.model = original_model
        self.output_index = output_index  # Index of the desired output tensor

    def forward(self, input_tensor):
        assert input_tensor.size(1) % 2 == 0, "The channel dimension must be even to split into two inputs."

        C = input_tensor.size(1) // 2
        image1 = input_tensor[:, :C, :, :]  # First half of channels
        image2 = input_tensor[:, C:, :, :]  # Second half of channels
        outputs = self.model(image1, image2)

        if isinstance(outputs, tuple) or isinstance(outputs, list):
            return outputs[self.output_index]
        return outputs
    

class SingleViewMamba(nn.Module):
    def __init__(self, 
                 in_channels, 
                 outputs,
                 pretrained=None):
        super().__init__()
        
        assert in_channels == 1, 'in_channels expected to be 1'
        self.in_channels = in_channels
        self.outputs = outputs
        
        model = build_model(num_classes=self.outputs)
        if pretrained is not None:
            checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            state_dict = checkpoint['model']
            excluded_prefix = 'classifier.head.'
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(excluded_prefix)}
            
            incompatible_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Successfully loaded checkpoint from {pretrained}")
            
            if incompatible_keys.unexpected_keys:
                print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
            if incompatible_keys.missing_keys:
                print(f"Missing keys: {incompatible_keys.missing_keys}")
        
        self.singleviewmamba = model
    
    def forward(self, x):
        # from 1 channel to 3
        x = x.expand(-1, 3, -1, -1)
        y = self.singleviewmamba(x)
        
        return y
    
class TwoViewLateJoinMamba(nn.Module):
    def __init__(self, in_channels, outputs, hidden_dim=768*2, pretrained=None
                ):
        super().__init__()
        
        assert in_channels == 1, 'in_channels expected to be 1'
        
        ################################### 1, feature extraction ###################################
        self.mamba_feature_extrac= Backbone_VSSM(pretrained=pretrained)
        
        ################################### 3, classifier ######################################
        self.classifier = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(hidden_dim, outputs),
        ))
        
    def forward(self, x_a, x_b):
        # from 1 channel to 3
        x_a = x_a.expand(-1, 3, -1, -1)
        x_b = x_b.expand(-1, 3, -1, -1)
        
        z_a = self.mamba_feature_extrac(x_a)
        z_b = self.mamba_feature_extrac(x_b)
        
        z_a = z_a[3]
        z_b = z_b[3]
        
        z_fuse = torch.cat([z_a, z_b], dim=1)
        
        y = self.classifier (z_fuse)
        
        return y
 
class TwoViewEarlyFusionMamba(nn.Module):
    def __init__(self, in_channels, outputs, hidden_dim=768, pretrained=None
                ):
        super().__init__()
        
        assert in_channels == 1, 'in_channels expected to be 1'
        
        ################################### 1, feature extraction ###################################
        self.mamba_feature_extrac = Backbone_VSSM(pretrained=pretrained)
        
        ################################### 2, early fusion ###################################
        self.early_fusion = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1),  # Fuse 3+3 channels after expansion
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        ################################### 3, classifier ######################################
        self.classifier = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(hidden_dim, outputs),
        ))
        
    def forward(self, x_a, x_b):
        # from 1 channel to 3
        x_a = x_a.expand(-1, 3, -1, -1)
        x_b = x_b.expand(-1, 3, -1, -1)
        
        # Early fusion
        x_fuse = torch.cat([x_a, x_b], dim=1)
        x_fuse = self.early_fusion(x_fuse)
        
        # Feature extraction
        z_fuse = self.mamba_feature_extrac(x_fuse)
        z_fuse = z_fuse[3]
        
        # Classification
        y = self.classifier(z_fuse)
        
        return y


class TwoViewXFMambaTop(nn.Module):
    def __init__(self, in_channels, outputs, attention_downsampling=4,
                 hidden_dim=768, depth=1, attn_drop_rate=0., d_state=16, drop_path_rate = 0.1,
                 pretrained=None, type='small'
                ):
        super().__init__()

        assert in_channels == 1, 'in_channels expected to be 1'
            
        ################################### 1, feature extraction ###################################
        if type == 'small':
            self.mamba_feature_extrac= Backbone_VSSM(depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, ssm_ratio=2.0, 
                                                    pretrained=pretrained)
        elif type == 'base':
            self.mamba_feature_extrac= Backbone_VSSM(depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6, ssm_ratio=2.0, 
                                                    pretrained=pretrained)
        elif type == 'tiny':
            self.mamba_feature_extrac= Backbone_VSSM(depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, ssm_ratio=1.0, 
                                                    pretrained=pretrained)
        
        ################################### 2, shallow feature fusion ######################################
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.shallow_mamba_fusion = ShallowFusionBlock_v4(
            hidden_dim=hidden_dim,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
        )
        
        ################################### 3, deep feature fusion ######################################
        self.fusemamba = CSSFVSSLayer_v5(
            hidden_dim=hidden_dim,
            depth=1,
            drop_path=dpr,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            attention_downsampling=attention_downsampling
        )
        
        ################################### 4, classifier ######################################
        self.final_conv = nn.Conv2d(
            in_channels=hidden_dim,  
            out_channels=hidden_dim,  
            kernel_size=1
        )
        
        self.classifier = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(hidden_dim, outputs),
        ))
        
    def forward(self, x_a, x_b):
        # from 1 channel to 3
        x_a = x_a.expand(-1, 3, -1, -1)
        x_b = x_b.expand(-1, 3, -1, -1)
        
        z_a = self.mamba_feature_extrac(x_a)
        z_b = self.mamba_feature_extrac(x_b)
        
        z_a = z_a[3]
        z_b = z_b[3] 
        
        z_a_shallow, z_b_shallow = self.shallow_mamba_fusion(z_a, z_b)
          
        z_fuse = self.fusemamba(z_a_shallow, z_b_shallow)
        
        z_fuse = self.final_conv(z_fuse)
        y = self.classifier (z_fuse)
        
        return y



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # x1 = torch.randn(1, 1, 224, 224).to(device)  
    # x2 = torch.randn(1, 1, 224, 224).to(device)  
    # model = TwoViewCrossFusionMambav13Top(in_channels=1, outputs=2).to(device)  
    # y = model(x1, x2)
    # print(y)
    # summary(model, input_size=[(1, 1, 224, 224), (1, 1, 224, 224)], device=device.type)
    x1 = torch.randn(1, 1, 224, 224).to(device) 
    model = SingleViewMamba(in_channels=1, outputs=2).to(device)  
    y = model(x1)
    print(model)
    summary(model, input_size=[(1, 1, 224, 224)], device=device.type)