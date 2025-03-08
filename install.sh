#!/bin/bash

# Fungsi untuk memeriksa dan memasang dependencies
check_dependencies() {
    echo "Memeriksa dependencies..."
    
    # Periksa Python 3.9+
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 tidak dijumpai. Sila pasang Python 3.9 atau lebih baharu."
        exit 1
    fi

    # Periksa pip
    if ! command -v pip3 &> /dev/null; then
        echo "pip3 tidak dijumpai. Memasang pip3..."
        sudo rpm-ostree install python3-pip
    fi

    # Periksa Vulkan
    if ! command -v vulkaninfo &> /dev/null; then
        echo "Vulkan tidak dijumpai. Memasang Vulkan..."
        sudo rpm-ostree install vulkan-tools mesa-vulkan-drivers
    fi
}

# Fungsi untuk memasang dependencies Python
install_python_deps() {
    echo "Memasang dependencies Python..."
    pip3 install -r requirements.txt
}

# Fungsi untuk setup direktori
setup_directories() {
    echo "Menyediakan direktori..."
    
    # Buat direktori konfigurasi
    sudo mkdir -p /etc/universal-ai-driver
    sudo cp config/config.yaml /etc/universal-ai-driver/

    # Buat direktori untuk executable
    sudo mkdir -p /usr/local/bin/universal-ai-driver
    sudo cp -r src/* /usr/local/bin/universal-ai-driver/

    # Buat symlink untuk executable utama
    sudo ln -sf /usr/local/bin/universal-ai-driver/core/ai_manager.py /usr/local/bin/universal-ai-driver
    sudo chmod +x /usr/local/bin/universal-ai-driver
}

# Fungsi untuk setup systemd service
setup_service() {
    echo "Menyediakan systemd service..."
    
    # Salin fail service
    sudo cp systemd/universal-ai-driver.service /etc/systemd/system/
    
    # Reload daemon dan aktifkan service
    sudo systemctl daemon-reload
    sudo systemctl enable universal-ai-driver.service
}

# Fungsi utama
main() {
    echo "Memulakan pemasangan Universal AI Driver..."
    
    check_dependencies
    install_python_deps
    setup_directories
    setup_service
    
    echo "Pemasangan selesai!"
    echo "Sila reboot sistem anda untuk memulakan Universal AI Driver."
}

# Jalankan fungsi utama
main 