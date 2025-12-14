"""
Diabetic Retinopathy Detection - Streamlit App
GDGOC PIEAS AI/ML Hackathon 2025

Run with: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

# Page config
st.set_page_config(
    page_title="DR Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .severity-0 { background-color: #d4edda; border-left: 5px solid #28a745; }
    .severity-1 { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .severity-2 { background-color: #fff3cd; border-left: 5px solid #fd7e14; }
    .severity-3 { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .severity-4 { background-color: #f8d7da; border-left: 5px solid #721c24; }
    
    /* Compact spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        padding-bottom: 0.5rem;
    }
    h2 {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        font-size: 1.3rem;
    }
    .stProgress > div > div > div > div {
        height: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture (same as training)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.bn(self.conv(x_cat)))
        return x * attention

class ImprovedDRModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.ca1 = ChannelAttention(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.ca2 = ChannelAttention(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.sa = SpatialAttention()
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.ca4 = ChannelAttention(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.ca1(x)
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.ca2(x)
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.sa(x)
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.ca4(x)
        x = self.pool(x)
        
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Grad-CAM for visualization (FIXED VERSION)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks and store them
        hook1 = target_layer.register_forward_hook(self.save_activation)
        hook2 = target_layer.register_full_backward_hook(self.save_gradient)
        self.hooks.append(hook1)
        self.hooks.append(hook2)
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def generate(self, input_tensor):
        # CRITICAL: Enable gradients for this tensor
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Set model to train mode to enable gradients
        self.model.train()
        
        # Forward pass
        output = self.model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, pred_idx].backward()
        
        # Check if gradients were captured
        if self.gradients is None:
            raise ValueError("Gradients not captured. Check hook registration.")
        
        # Process Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Set model back to eval mode
        self.model.eval()
        
        return cam, pred_idx, F.softmax(output, dim=1)[0].cpu().detach().numpy()
    
    def remove_hooks(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Load model
@st.cache_resource
def load_model(use_quantized=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedDRModel(num_classes=5)
    
    if use_quantized:
        # Load quantized model (CPU only)
        checkpoint = torch.load('model/quantized_model.pt', map_location='cpu', weights_only=False)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, torch.device('cpu')
    else:
        # Load fine-tuned model
        try:
            checkpoint = torch.load('model/finetuned_model.pt', map_location=device, weights_only=False)
        except FileNotFoundError:
            checkpoint = torch.load('model/best_model.pt', map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Main app
def main():
    st.markdown('<h1 class="main-header">🏥 Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">GDGOC PIEAS AI/ML Hackathon 2025 | AI-Powered Early Diagnosis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        AI model for diabetic retinopathy detection.
        
        **Performance:**
        - Accuracy: 70.86%
        - Severe DR: 84.00%
        - Proliferative: 90.19%
        
        **Optimizations:**
        - Fine-tuned model
        - CPU-optimized
        - 9.5ms inference
        """)
        
        st.header("⚙️ Settings")
        use_quantized = st.checkbox("Use Quantized Model", value=False, 
                                    help="Faster inference, minimal accuracy drop")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        show_probabilities = st.checkbox("Show Probabilities", value=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Retinal Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a color fundus photograph of the retina"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width=400)
    
    with col2:
        if uploaded_file is not None:
            st.header("🔍 Analysis Results")
            
            with st.spinner("Analyzing image..."):
                # Load model based on user selection
                model, device = load_model(use_quantized=use_quantized)
                
                # Preprocess
                input_tensor = preprocess_image(image).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
                    predicted_class = probabilities.argmax()
                
                class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
                confidence = probabilities[predicted_class]
                
                # Display result
                st.markdown(f"### **{class_names[predicted_class]}**")
                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                if use_quantized:
                    st.caption("⚡ Using quantized model (faster)")
                else:
                    st.caption("🎯 Using fine-tuned model (best accuracy)")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommendations
                if predicted_class == 0:
                    st.success("✅ No DR detected. Continue regular screenings.")
                elif predicted_class == 1:
                    st.warning("⚠️ Mild DR. Consult ophthalmologist.")
                elif predicted_class == 2:
                    st.warning("⚠️ Moderate DR. Regular follow-ups needed.")
                elif predicted_class == 3:
                    st.error("🚨 Severe DR. Medical consultation required.")
                else:
                    st.error("🚨 Proliferative DR. Urgent attention needed!")
                
                # Show probabilities
                if show_probabilities:
                    st.markdown("**📊 Probabilities:**")
                    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")
    
    # Grad-CAM visualization (FIXED VERSION)
    if uploaded_file is not None and show_gradcam:
        st.header("🎯 Attention Heatmap (Grad-CAM)")
        
        with st.spinner("Generating heatmap..."):
            try:
                # Reload model for Grad-CAM (needs gradients enabled)
                gradcam_model, gradcam_device = load_model(use_quantized=False)
                gradcam_input = preprocess_image(image).to(gradcam_device)
                
                gradcam = GradCAM(gradcam_model, gradcam_model.conv5)
                cam, _, _ = gradcam.generate(gradcam_input)
                gradcam.remove_hooks()  # Clean up
                
                # Create compact visualization
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
                
                # Original
                axes[0].imshow(image.resize((224, 224)))
                axes[0].set_title("Original", fontsize=11)
                axes[0].axis('off')
                
                # Heatmap
                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title("Attention", fontsize=11)
                axes[1].axis('off')
                
                # Overlay
                img_array = np.array(image.resize((224, 224)))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
                
                axes[2].imshow(overlay)
                axes[2].set_title(f"Focus Areas", fontsize=11)
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                st.info("🔍 The heatmap shows which regions the AI focused on when making its diagnosis.")
            
            except Exception as e:
                st.error(f"Could not generate Grad-CAM: {str(e)}")
                st.info("Note: Grad-CAM requires the full model (not quantized)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85rem;'>
        <p>⚕️ <strong>Disclaimer:</strong> For screening only. Consult a healthcare professional.</p>
        <p>GDGOC PIEAS AI/ML Hackathon 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()