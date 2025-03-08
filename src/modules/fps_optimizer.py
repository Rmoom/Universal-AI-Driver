#!/usr/bin/env python3

import os
import subprocess
import logging
from typing import Dict, Optional
import psutil
import numpy as np

class FPSOptimizer:
    def __init__(self, config: dict):
        self.logger = logging.getLogger("FPSOptimizer")
        self.config = config
        self.optimization_active = False
        
    def apply_system_optimizations(self):
        """Aplikasi optimasi sistem untuk FPS maksimum"""
        try:
            # Set CPU governor ke performance
            self._set_cpu_governor("performance")
            
            # Matikan services yang tidak diperlukan
            self._disable_unnecessary_services()
            
            # Set process priority
            self._set_process_priority()
            
            # Optimasi memory
            self._optimize_memory()
            
            # Optimasi GPU
            self._optimize_gpu()
            
            self.optimization_active = True
            self.logger.info("Optimasi FPS telah diaktifkan")
            
        except Exception as e:
            self.logger.error(f"Gagal mengaplikasi optimasi: {e}")

    def _set_cpu_governor(self, governor: str):
        """Set CPU governor"""
        try:
            cmd = f"echo {governor} | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            self.logger.warning(f"Gagal set CPU governor: {e}")

    def _disable_unnecessary_services(self):
        """Matikan services yang tidak diperlukan"""
        services_to_stop = [
            "bluetooth", "cups", "avahi-daemon", 
            "ModemManager", "NetworkManager"
        ]
        
        for service in services_to_stop:
            try:
                cmd = f"sudo systemctl stop {service}"
                subprocess.run(cmd, shell=True, check=True)
            except Exception as e:
                self.logger.warning(f"Gagal hentikan service {service}: {e}")

    def _set_process_priority(self):
        """Set priority proses untuk gaming"""
        try:
            # Set process priority
            os.nice(-20)
            
            # Set CPU affinity untuk proses utama
            p = psutil.Process()
            p.cpu_affinity(list(range(psutil.cpu_count())))
            
        except Exception as e:
            self.logger.warning(f"Gagal set process priority: {e}")

    def _optimize_memory(self):
        """Optimasi penggunaan memory"""
        try:
            # Clear disk cache
            subprocess.run("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches", 
                         shell=True, check=True)
            
            # Set swappiness rendah
            subprocess.run("echo 10 | sudo tee /proc/sys/vm/swappiness", 
                         shell=True, check=True)
            
            # Aktifkan Transparent Huge Pages
            subprocess.run("echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled", 
                         shell=True, check=True)
            
        except Exception as e:
            self.logger.warning(f"Gagal optimasi memory: {e}")

    def _optimize_gpu(self):
        """Optimasi tetapan GPU"""
        try:
            # Set power limit lebih tinggi untuk AMD GPU
            if self.config['gpu']['amd_dgpu']['enabled']:
                cmd = "echo high | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level"
                subprocess.run(cmd, shell=True, check=True)
            
            # Kurangkan beban pada Intel iGPU
            if self.config['gpu']['intel_igpu']['enabled']:
                cmd = "echo powersave | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level"
                subprocess.run(cmd, shell=True, check=True)
                
        except Exception as e:
            self.logger.warning(f"Gagal optimasi GPU: {e}")

    def monitor_fps(self) -> Dict:
        """Pantau FPS semasa dan beri cadangan"""
        try:
            # TODO: Implement actual FPS monitoring
            current_fps = 60  # Dummy value
            
            recommendations = []
            if current_fps < self.config['ai']['frame_generation']['target_fps']:
                if current_fps < 30:
                    recommendations.append("Kurangkan resolusi ke 480p")
                    recommendations.append("Matikan semua effects")
                elif current_fps < 60:
                    recommendations.append("Kurangkan texture quality")
                    recommendations.append("Aktifkan aggressive frame generation")
                else:
                    recommendations.append("Fine-tune shader settings")
            
            return {
                'current_fps': current_fps,
                'target_fps': self.config['ai']['frame_generation']['target_fps'],
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Gagal monitor FPS: {e}")
            return {}

    def revert_optimizations(self):
        """Pulihkan tetapan asal"""
        try:
            if not self.optimization_active:
                return
                
            # Pulihkan CPU governor
            self._set_cpu_governor("ondemand")
            
            # Start semula services
            for service in ["bluetooth", "cups", "avahi-daemon", 
                          "ModemManager", "NetworkManager"]:
                subprocess.run(f"sudo systemctl start {service}", 
                             shell=True, check=True)
            
            # Pulihkan process priority
            os.nice(0)
            
            # Pulihkan memory settings
            subprocess.run("echo 60 | sudo tee /proc/sys/vm/swappiness", 
                         shell=True, check=True)
            
            self.optimization_active = False
            self.logger.info("Optimasi FPS telah dinyahaktifkan")
            
        except Exception as e:
            self.logger.error(f"Gagal pulihkan tetapan: {e}")

    def get_performance_stats(self) -> Dict:
        """Dapatkan statistik prestasi"""
        try:
            return {
                'cpu_usage': psutil.cpu_percent(interval=0.1),
                'memory_usage': psutil.virtual_memory().percent,
                'gpu_temp': self._get_gpu_temp(),
                'optimization_active': self.optimization_active
            }
        except Exception as e:
            self.logger.error(f"Gagal dapatkan stats: {e}")
            return {} 