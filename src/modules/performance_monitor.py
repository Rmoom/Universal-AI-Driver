#!/usr/bin/env python3

import psutil
import gputil
import time
import logging
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import threading

class PerformanceMonitor:
    def __init__(self, config: dict):
        self.logger = logging.getLogger("PerformanceMonitor")
        self.config = config
        self.sampling_rate = config['ai']['performance_monitoring']['sampling_rate']
        self.history_length = 100  # Simpan 100 sampel terakhir
        
        # Inisialisasi buffer untuk data
        self.fps_history = deque(maxlen=self.history_length)
        self.temp_history = deque(maxlen=self.history_length)
        self.gpu_usage_history = deque(maxlen=self.history_length)
        self.cpu_usage_history = deque(maxlen=self.history_length)
        
        self.running = False
        self.thread = None

    def start_monitoring(self):
        """Mulakan pemantauan prestasi"""
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        """Hentikan pemantauan prestasi"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitoring_loop(self):
        """Loop utama untuk pemantauan prestasi"""
        while self.running:
            try:
                # Kumpul metrik prestasi
                metrics = self._collect_metrics()
                
                # Simpan dalam history
                self.fps_history.append(metrics['fps'])
                self.temp_history.append(metrics['gpu_temp'])
                self.gpu_usage_history.append(metrics['gpu_usage'])
                self.cpu_usage_history.append(metrics['cpu_usage'])
                
                # Analisis prestasi
                self._analyze_performance(metrics)
                
                # Tunggu sehingga sampling rate seterusnya
                time.sleep(self.sampling_rate / 1000)  # Convert ke saat
                
            except Exception as e:
                self.logger.error(f"Ralat dalam loop pemantauan: {e}")
                time.sleep(1)  # Tunggu sebentar sebelum cuba semula

    def _collect_metrics(self) -> Dict:
        """Kumpul metrik prestasi sistem"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # GPU metrics (AMD)
            gpus = gputil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Andaikan GPU pertama adalah AMD
                gpu_temp = gpu.temperature
                gpu_usage = gpu.load * 100
            else:
                gpu_temp = 0
                gpu_usage = 0
            
            # FPS estimation (dummy untuk sekarang)
            # TODO: Implement actual FPS monitoring
            fps = 60
            
            return {
                'cpu_usage': cpu_usage,
                'gpu_usage': gpu_usage,
                'gpu_temp': gpu_temp,
                'fps': fps,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Gagal mengumpul metrik: {e}")
            return {
                'cpu_usage': 0,
                'gpu_usage': 0,
                'gpu_temp': 0,
                'fps': 0,
                'timestamp': time.time()
            }

    def _analyze_performance(self, metrics: Dict):
        """Analisis prestasi dan buat keputusan"""
        try:
            # Periksa suhu
            if metrics['gpu_temp'] > self.config['ai']['performance_monitoring']['temperature_limit']:
                self.logger.warning(f"Suhu GPU tinggi: {metrics['gpu_temp']}°C")
                self._handle_high_temperature()
            
            # Periksa FPS
            target_fps = self.config['ai']['performance_monitoring']['fps_target']
            if metrics['fps'] < target_fps * 0.8:  # 20% di bawah target
                self.logger.warning(f"FPS rendah: {metrics['fps']}")
                self._handle_low_fps()
            
            # Periksa penggunaan GPU
            if metrics['gpu_usage'] > 95:  # GPU usage sangat tinggi
                self.logger.warning(f"Penggunaan GPU tinggi: {metrics['gpu_usage']}%")
                self._handle_high_gpu_usage()
                
        except Exception as e:
            self.logger.error(f"Gagal menganalisis prestasi: {e}")

    def _handle_high_temperature(self):
        """Handle suhu tinggi"""
        # TODO: Implement temperature management
        # Contoh: Kurangkan clock speed, tingkatkan fan speed
        pass

    def _handle_low_fps(self):
        """Handle FPS rendah"""
        # TODO: Implement FPS optimization
        # Contoh: Aktifkan upscaling, kurangkan kualiti
        pass

    def _handle_high_gpu_usage(self):
        """Handle penggunaan GPU tinggi"""
        # TODO: Implement GPU usage optimization
        # Contoh: Offload sebahagian kerja ke iGPU
        pass

    def get_performance_report(self) -> Dict:
        """Dapatkan laporan prestasi semasa"""
        try:
            return {
                'fps_avg': np.mean(self.fps_history) if self.fps_history else 0,
                'fps_min': np.min(self.fps_history) if self.fps_history else 0,
                'gpu_temp_avg': np.mean(self.temp_history) if self.temp_history else 0,
                'gpu_usage_avg': np.mean(self.gpu_usage_history) if self.gpu_usage_history else 0,
                'cpu_usage_avg': np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0
            }
        except Exception as e:
            self.logger.error(f"Gagal menjana laporan prestasi: {e}")
            return {} 