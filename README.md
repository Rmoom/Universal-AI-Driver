# Universal AI Driver untuk Linux Silverblue

Sistem pemandu AI yang dioptimumkan untuk Linux Silverblue yang menggunakan Intel iGPU untuk pemprosesan AI dan AMD dGPU untuk rendering utama.

## 🚀 Ciri-ciri Utama

- AI Adaptive Performance Monitoring
- AI Auto-Upscaling & Image Enhancement
- AI Auto-Frame Generation
- AI Adaptive Latency Reduction
- Hybrid GPU Processing (Intel iGPU + AMD dGPU)
- Pembelajaran Mesin Berterusan

## 📋 Keperluan Sistem

- Linux Silverblue Fedora (versi terkini)
- Intel iGPU (Gen 6+)
- AMD dGPU (R5 M330 atau lebih baharu)
- Minimum 4GB RAM
- Python 3.7+
- Vulkan 1.1+

## 🛠️ Teknologi yang Digunakan

- OpenVINO untuk AI processing pada Intel iGPU
- ROCm untuk AMD GPU acceleration
- Vulkan API untuk rendering
- TensorFlow Lite & PyTorch untuk pembelajaran mesin
- systemd untuk perkhidmatan sistem

## 📦 Cara Pemasangan

```bash
# 1. Pasang dependencies yang diperlukan
rpm-ostree install python3-pip vulkan-tools mesa-vulkan-drivers

# 2. Pasang OpenVINO dan ROCm
# OpenVINO
sudo rpm-ostree override replace --experimental --install intel-openvino
toolbox create --distro fedora --release 41  
toolbox enter fedora-41  
sudo dnf install intel-openvino
source /opt/intel/openvino/setupvars.sh
echo 'source /opt/intel/openvino/setupvars.sh' >> ~/.bashrc

# ROCm
sudo rpm-ostree override replace --experimental --install rocm-opencl rocm-opencl-devel rocm-smi
toolbox create --distro fedora --release 39  
toolbox enter fedora-39  
sudo dnf install rocm-opencl rocm-opencl-devel rocm-smi
sudo usermod -aG video $USER  #Kemudian reboot sistem.
rocm-smi  #Selepas reboot, jalankan

# 3. Clone repo ini
git clone https://github.com/Rmoom/universal-ai-driver

# 4. Jalankan script pemasangan
cd universal-ai-driver
./install.sh
```

## 🔧 Konfigurasi

Fail konfigurasi terletak di:
```
/etc/universal-ai-driver/config.yaml
```

## 📊 Prestasi yang Dijangka

- Peningkatan FPS: 15-30%
- Pengurangan input lag: ~10ms
- Peningkatan kualiti imej: Medium
- Penggunaan RAM: ~500MB
- Penggunaan GPU: 5-15% (iGPU) + 60-80% (dGPU)

## 📊 Status Projek

🚧 Dalam Pembangunan 🚧

## 📝 Lesen

MIT License 
