import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from matplotlib.patches import Circle
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import pygame
from copy import deepcopy
from typing import Dict, List , Any
import multiprocessing
import torch.cuda.amp
import traceback
import os
from tqdm import tqdm
import math
import copy
import time
from datetime import timedelta
# GPU kontrolü için en başa ekleyin
print("GPU Durumu:")
print(f"PyTorch: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
   print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
   print(f"GPU Sayısı: {torch.cuda.device_count()}")
   print(f"Mevcut GPU İndeksi: {torch.cuda.current_device()}")
else:
   print("GPU bulunamadı, CPU kullanılacak")

# Görselleştirme modu kontrolü
VISUALIZATION_MODE = False  # True: Test/Geliştirme modu, False: Eğitim modu

# Sabitler
MAX_WORKERS = 48 if not VISUALIZATION_MODE else 1  # Eğitim modunda 48, görselleştirme modunda 1 işçi
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# CUDA yapılandırması ve optimizasyonları
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA hata ayıklama için
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Bellek parçalanmasını önle
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # TF32'yi etkinleştir
torch.backends.cudnn.allow_tf32 = True        # cuDNN TF32'yi etkinleştir
torch.cuda.empty_cache()

# Fiziksel sabitler
G = 6.67430e-11          # Gravitasyon sabiti (m³/kg/s²)
SUN_MASS = 1.989e30      # Güneş kütlesi (kg)
AU = 1.496e11            # Astronomik birim (m)
TOTAL_TIME = 15 * 365 * 24 * 3600  # Toplam simülasyon süresi (15 yıl)
TIME_STEP = 86400        # Zaman adımı (1 gün)
NUM_STEPS = int(TOTAL_TIME / TIME_STEP)
MAX_THRUST = 1e3         # Maksimum itki kuvveti (N)
ISP = 4000              # Özgül impuls (s)
g0 = 9.81               # Yerçekimi ivmesi (m/s²)
SOLAR_SYSTEM_BOUNDARY = 60 * AU  # 60 AU'luk görüş alanı
SPEED_OF_LIGHT = 3e8    # Işık hızı (m/s)
MAX_SPACECRAFT_SPEED = 150000  # Maksimum uzay aracı hızı (m/s)
OPTIMAL_VELOCITY = 40000  # Optimal hız (40 km/s)
MAX_ACCELERATION = 0.1   # Maksimum ivme (m/s²)
MIN_SAFE_DISTANCE_SUN = 696340000 * 2  # Güneş'e minimum güvenli mesafe (2 güneş yarıçapı)
SUCCESS_DISTANCE = 5e6   # Başarılı varış mesafesi (m)
MAX_EPISODE_TIME = 15* 365 * 24 * 3600  # Maksimum bölüm süresi (15 yıl)

# Acil durum olasılıkları
MOTOR_FAILURE_PROBABILITY = 0.00001      # Motor arıza olasılığı
FUEL_LEAK_PROBABILITY = 0.00002          # Yakıt sızıntısı olasılığı
MICROASTEROID_COLLISION_PROBABILITY = 0.0001  # Mikroasteroit çarpışma olasılığı
NUM_EPISODES = 50000    # Toplam eğitim bölümü sayısı
MAX_STEPS = 50000       # Bir bölümdeki maksimum adım sayısı
 
# Görelilik fiziği hesaplamalarını yapan sınıf
class RelativisticSpacePhysics:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.c = 299792458.0  # Işık hızı (m/s)
        self.G = 6.67430e-11  # Gravitasyon sabiti (m³/kg/s²)
        self.sun_mass = 1.989e30  # Güneş kütlesi (kg)
        
        # Schwarzschild yarıçapı - Kara deliğin olay ufku yarıçapı
        self.rs = 2 * self.G * self.sun_mass / (self.c ** 2)
        
    def schwarzschild_metric(self, r):
        """Schwarzschild metriğini hesapla - Kara delik etrafındaki uzay-zaman eğriliğini tanımlar"""
        try:
            # Metrik bileşenleri
            g_tt = -(1 - self.rs / r)  # Zamansal bileşen - Zaman genişlemesi
            g_rr = 1 / (1 - self.rs / r)  # Radyal bileşen - Uzaysal eğrilik
            g_thth = r ** 2  # Açısal bileşen (theta) - Küresel koordinatlar
            g_phph = r ** 2 * torch.sin(torch.tensor(torch.pi/2, device=self.device)) ** 2  # Açısal bileşen (phi)
            
            return g_tt, g_rr, g_thth, g_phph
        except Exception as e:
            print(f"Metrik hesaplama hatası: {e}")
            return None
        
    def calculate_relativistic_force(self, position, velocity, mass):
        """Görelilik etkilerini içeren kuvveti hesapla - Özel ve genel görelilik etkilerini birleştirir"""
        try:
            r = torch.norm(position)  # Güneş'e olan uzaklık
            v = torch.norm(velocity)  # Hız büyüklüğü
            
            # Klasik Newton kuvveti
            F_classical = -self.G * self.sun_mass * mass * position / (r**3)
            
            # Görelilik düzeltmeleri
            gamma = 1.0 / torch.sqrt(1.0 - (v/self.c)**2)  # Lorentz faktörü
            
            # Schwarzschild metriği - Uzay-zaman eğriliği
            metric_result = self.schwarzschild_metric(r)
            if metric_result is None:
                return F_classical # Hata durumunda klasik kuvveti döndür
                
            g_tt, g_rr, _, _ = metric_result
            
            # Görelilik düzeltmesi faktörleri
            perihelion_shift = 1 + 3 * self.G * self.sun_mass / (r * self.c**2)  # Merkür'ün perihel kayması
            time_dilation = torch.sqrt(-g_tt)  # Zaman genişlemesi
            spatial_curvature = torch.sqrt(g_rr)  # Uzaysal eğrilik
            
            # Toplam görelilik etkisi
            relativistic_correction = gamma * perihelion_shift * time_dilation * spatial_curvature
            
            # Düzeltilmiş kuvvet
            F_relativistic = F_classical * relativistic_correction
            
            # Kuvvet sınırlaması - Işık hızı limiti
            max_force = mass * self.c**2 / r  # Maksimum izin verilen kuvvet
            F_magnitude = torch.norm(F_relativistic)
            if F_magnitude > max_force:
                F_relativistic = F_relativistic * (max_force / F_magnitude)
            
            return F_relativistic
                
        except Exception as e:
            print(f"Görelilik kuvveti hesaplama hatası: {e}")
            return -self.G * self.sun_mass * mass * position / (r**3)  # Hata durumunda klasik kuvveti döndür
        
    def update_planet_motion(self, planet, dt):
        """Gezegen hareketini Runge-Kutta-Fehlberg (RKF45) metodu ile güncelle"""
        try:
            # Görelilik etkili kuvvet
            force = self.calculate_relativistic_force(planet.position, planet.velocity, planet.mass)
            
            # Lorentz faktörü - Özel görelilik düzeltmesi
            v = torch.norm(planet.velocity)
            gamma = 1 / torch.sqrt(1 - (v/self.c)**2)
            
            # Görelilik etkili ivme (F = γma)
            self.acceleration = force / (gamma * planet.mass)
            
            # RKF45 katsayıları
            a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1, 1/2
            b21 = 1/4
            b31, b32 = 3/32, 9/32
            b41, b42, b43 = 1932/2197, -7200/2197, 7296/2197
            b51, b52, b53, b54 = 439/216, -8, 3680/513, -845/4104
            b61, b62, b63, b64, b65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
            
            # RKF45 adımları
            def f(t, y):
                pos, vel = y[:2], y[2:]
                acc = self.calculate_relativistic_force(pos, vel, planet.mass) / (gamma * planet.mass)
                return torch.cat([vel, acc])
            
            y = torch.cat([planet.position, planet.velocity])
            t = torch.tensor(0.0, device=self.device)
            
            # k1 hesapla
            k1 = dt * f(t, y)
            
            # k2 hesapla
            k2 = dt * f(t + a2*dt, y + b21*k1)
            
            # k3 hesapla
            k3 = dt * f(t + a3*dt, y + b31*k1 + b32*k2)
            
            # k4 hesapla
            k4 = dt * f(t + a4*dt, y + b41*k1 + b42*k2 + b43*k3)
            
            # k5 hesapla
            k5 = dt * f(t + a5*dt, y + b51*k1 + b52*k2 + b53*k3 + b54*k4)
            
            # k6 hesapla
            k6 = dt * f(t + a6*dt, y + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5)
            
            # 4. derece çözüm
            y4 = y + (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5
            
            # 5. derece çözüm
            y5 = y + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
            
            # Hata tahmini
            error = torch.norm(y5 - y4)
            
            # Optimal adım boyutu hesaplama
            tolerance = 1e-6
            optimal_dt = dt * (tolerance / (2 * error))**0.2
            
            # Eğer hata çok büyükse adım boyutunu küçült
            if error > tolerance:
                return self.update_planet_motion(planet, optimal_dt)
            
            # Yeni pozisyon ve hızı ayarla
            new_position = y5[:2]
            new_velocity = y5[2:]
            
            # Hız sınırlaması - Işık hızını geçemez
            v_new = torch.norm(new_velocity)
            if v_new >= self.c:
                new_velocity = new_velocity * (0.99 * self.c / v_new)
            
            return new_position, new_velocity
                
        except Exception as e:
            print(f"Hareket güncelleme hatası: {e}")
            return planet.position, planet.velocity
        
    def calculate_light_bending(self, impact_parameter):
        """Işık sapmasını hesapla - Einstein'ın genel görelilik öngörüsü"""
        try:
            deflection_angle = 4 * self.G * self.sun_mass / (impact_parameter * self.c**2)
            return deflection_angle
        except Exception as e:
            print(f"Işık sapması hesaplama hatası: {e}")
            return 0.0
        
    def gravitational_redshift(self, r):
        """Gravitasyonel kırmızıya kaymayı hesapla - Güçlü gravitasyon alanlarında ışığın dalga boyundaki değişim"""
        try:
            redshift = 1 / torch.sqrt(1 - self.rs/r) - 1
            return redshift
        except Exception as e:
            print(f"Kırmızıya kayma hesaplama hatası: {e}")
            return 0.0

# Genetik algoritma ve Transformer tabanlı optimizasyon sınıfı
class GeneticTransformerOptimizer:
    def __init__(self, 
                 population_size=48,      # Popülasyon büyüklüğü
                 num_generations=1000,     # Nesil sayısı
                 mutation_rate=0.1,        # Mutasyon oranı
                 d_model=64,              # Transformer model boyutu
                 n_head=4,                # Attention head sayısı
                 n_layers=4,              # Transformer katman sayısı
                 d_ff=512,                # Feed-forward boyutu
                 device='cuda'):
        
        self.device = device
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        
        # Transformer model parametreleri
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.d_ff = d_ff
        
        # Model oluşturma - SAC (Soft Actor-Critic) ağı
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_ff,
                dropout=0.1,
                batch_first=True
            ).to(device),
            num_layers=n_layers
        ).to(device)
        
        # Optimizer ve gradient scaler
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=1e-4)
        self.scaler =torch.amp.GradScaler('cuda')
        
    def create_population(self):
        """Başlangıç popülasyonunu oluştur - 360 derecelik yörünge noktaları"""
        return torch.randn(self.population_size, 360, 2, device=self.device)
    
    def evaluate_population_batch(self, population_batch, env):
        """GPU üzerinde toplu değerlendirme - Paralel simülasyon"""
        batch_size = len(population_batch)
        rewards = torch.zeros(batch_size, device=self.device)
        
        states = torch.stack([env.reset() for _ in range(batch_size)])
        dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                # Transformer girişi için boyut düzeltmesi
                states_input = states.unsqueeze(1).float()  # [batch_size, 1, state_dim]
                actions = self.transformer(states_input).squeeze(1)  # [batch_size, action_dim]
            
            # Çevreye gönderilecek aksiyonların tipini kontrol et
            actions = actions.detach()  # Gradyanları kopar
            
            # Batch işlemi için tip kontrolü
            next_states, step_rewards, step_dones, _ = env.step_batch(actions)
            
            # Ödül ve done maskelerinin tiplerini kontrol et 
            step_rewards = step_rewards.to(self.device)
            step_dones = step_dones.to(self.device)
            
            # Ödül ve done güncellemeleri
            rewards += step_rewards * (~dones)  
            dones |= step_dones
            
            if dones.all():
                break
                
            states = next_states.clone()  # Güvenli kopya
            
        return rewards
    
    def crossover(self, parent1, parent2):
        """İki ebeveyn arasında çaprazlama yap"""
        try:
            # Yeni bir çocuk oluştur
            child = deepcopy(parent1)
            
            # Her bir parametre için
            p1_params = dict(parent1.sac.named_parameters())
            p2_params = dict(parent2.sac.named_parameters())
            
            for name, param1 in p1_params.items():
                param2 = p2_params[name]
                
                # Boyut kontrolü yap
                if param1.data.shape != param2.data.shape:
                    print(f"Uyarı: {name} için boyut uyuşmazlığı - P1: {param1.data.shape}, P2: {param2.data.shape}")
                    continue  # Bu parametreyi atla
                    
                # Rastgele çaprazlama noktası seç
                if random.random() < self.crossover_rate:
                    mask = torch.rand_like(param1.data) < 0.5
                    child_param = torch.where(mask, param1.data, param2.data)
                    dict(child.sac.named_parameters())[name].data.copy_(child_param)
            
            return child
            
        except Exception as e:
            print(f"Çaprazlama hatası: {e}")
            return deepcopy(parent1)  # Hata durumunda parent1'i kopyala
    
    def mutate(self, trajectory):
        """Yörüngede mutasyon uygula - Gaussian gürültü ekleme"""
        mutation_mask = torch.rand_like(trajectory) < self.mutation_rate
        trajectory += mutation_mask * torch.randn_like(trajectory) * 0.1
        return trajectory
    
    def train(self, env):
        """Ana eğitim döngüsü - Genetik algoritma ve DQN kombinasyonu"""
        try:
            best_fitness = float('-inf')
            best_trajectory = None
            
            for generation in range(self.num_generations):
                # Yeni popülasyon oluştur
                population = self.create_population()
                fitness_scores = self.evaluate_population_batch(population, env)
                
                # En iyileri seç
                elite_indices = torch.topk(fitness_scores, k=self.population_size//4).indices
                elite = population[elite_indices]
                
                # Yeni nesil oluştur
                new_population = [elite]
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(elite.tolist(), 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)
                
                population = torch.stack(new_population)
                
                # En iyi sonucu güncelle
                max_fitness, max_idx = torch.max(fitness_scores, dim=0)
                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_trajectory = population[max_idx].clone()
                
                if generation % 10 == 0:
                    print(f"Nesil {generation}: En iyi fitness = {best_fitness:.2f}")
                    
        except Exception as e:
            print(f"Eğitim hatası: {e}")
            traceback.print_exc()
            
        return best_trajectory, best_fitness
    
    def physics_loss(self, state, next_state):
        """PINN fizik kaybını hesapla - Fizik yasalarına uygunluk kontrolü"""
        # Fizik kodlaması
        physics_encoding = self.current_model['physics_encoder'](state)
        next_physics_encoding = self.current_model['physics_encoder'](next_state)
        
        # Korunum yasaları kaybı
        conservation_pred = self.current_model['conservation_layer'](physics_encoding)
        next_conservation_pred = self.current_model['conservation_layer'](next_physics_encoding)
        
        # Enerji korunumu
        energy_loss = torch.mean((conservation_pred[:, 0] - next_conservation_pred[:, 0]) ** 2)
        
        # Momentum korunumu
        momentum_loss = torch.mean((conservation_pred[:, 1:] - next_conservation_pred[:, 1:]) ** 2)
        
        # Görelilik düzeltmeleri
        rel_pred = self.current_model['relativistic_layer'](physics_encoding)
        velocity = torch.norm(state[:, 3:5], dim=1)
        gamma = 1.0 / torch.sqrt(1.0 - (velocity/3e8)**2)
        rel_loss = torch.mean((rel_pred - gamma) ** 2)
        
        return energy_loss + momentum_loss + rel_loss

    @torch.amp.autocast('cuda')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri yayılım - Batch işleme ve bellek optimizasyonu"""
        batch_size = x.size(0)
        max_batch = 32  # Maksimum batch boyutu
        
        if batch_size <= max_batch:
            return self._forward_single(x)
        
        outputs = []
        for i in range(0, batch_size, max_batch):
            batch = x[i:i + max_batch]
            with torch.no_grad():
                output = self._forward_single(batch)
            outputs.append(output)
            
        return torch.cat(outputs, dim=0)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tek batch için ileri yayılım işlemi
        """
        try:
            # Giriş boyutu düzeltme ve veri tipi dönüşümü
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # [batch_size, 1, features]
            x = x.to(dtype=torch.float16)  # FP16'ya dönüştür
            
            # Giriş projeksiyonu
            x = self.current_model['input_proj'](x)  # [batch_size, seq_len, d_model]
            
            # Positional encoding ekle - güncellendi
            seq_len = x.size(1)
            pos_enc = self.current_model['pos_encoder'](
                torch.zeros(x.size(0), seq_len, dtype=torch.long, device=x.device))
            pos_enc = x + pos_enc
            
            # Transformer katmanı
            # Attention mask oluştur (False = herhangi bir padding yok)
            mask = torch.zeros((x.size(0), x.size(1)), device=x.device).bool()
            x = self.current_model['transformer'](x, src_key_padding_mask=mask)  # [batch_size, seq_len, d_model]
            
            # Global pooling - sequence boyutunu ortadan kaldır
            x = x.mean(dim=1)  # [batch_size, d_model]
            
            # Physics-Informed katmanlar
            with torch.amp.autocast('cuda'):
                # Fizik kodlaması
                physics_features = self.current_model['physics_encoder'](x)
                
                # Korunum yasaları
                conservation_pred = self.current_model['conservation_layer'](physics_features)
                conservation_factor = torch.sigmoid(conservation_pred[:, 0]).unsqueeze(1)
                
                # Görelilik etkileri
                relativistic_pred = self.current_model['relativistic_layer'](physics_features)
                gamma_factor = torch.sigmoid(relativistic_pred[:, 0]).unsqueeze(1)
            
                # SAC mimarisi
                actor_output = self.current_model['actor'](x)
                mean, log_std = torch.chunk(actor_output, 2, dim=-1)
                log_std = torch.clamp(log_std, min=-20, max=2)
                std = log_std.exp()
                
                # Gaussian dağılımdan örnekleme
                normal = torch.distributions.Normal(mean, std)
                action = normal.rsample()
                action = torch.tanh(action)
                
                # Log olasılığını hesapla
                log_prob = normal.log_prob(action)
                log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(1, keepdim=True)
            
            # Fizik tabanlı düzeltmeler
            physics_correction = (1.0 + 0.1 * conservation_factor) * (1.0 + 0.05 * gamma_factor)
            q_values = q_values * physics_correction
            
            # Çıktı normalizasyonu ve sınırlama
            q_values = torch.clamp(q_values, min=-100.0, max=100.0)
            
            # Bellek optimizasyonu
            del physics_features, conservation_pred, relativistic_pred
            del value_stream, advantage_stream, advantage_mean
            torch.cuda.empty_cache()
            
            return q_values,action, log_prob
            
        except Exception as e:
            print(f"Forward pass hatası: {str(e)}")
            print("Hata detayları:")
            print(f"Giriş boyutu: {x.shape}")
            print(f"Giriş tipi: {x.dtype}")
            print(f"Cihaz: {x.device}")
            traceback.print_exc()
            raise e
        
        finally:
            # Ek bellek temizliği
            if hasattr(self, '_temp_storage'):
                del self._temp_storage
            torch.cuda.empty_cache()

class LagrangePoints:
    def __init__(self, planet_mass, planet_orbit_radius):
        self.mu = planet_mass / (SUN_MASS + planet_mass)
        self.orbit_radius = planet_orbit_radius
        
        # L1 point
        self.l1_distance = self.orbit_radius * (1 - (self.mu/3)**(1/3))
        self.l1 = torch.tensor([self.l1_distance, 0.0], device=device)
        self.l2_distance = self.orbit_radius * (1 + (self.mu/3)**(1/3))
        self.l2 = torch.tensor([self.l2_distance, 0.0], device=device)
        self.influence_radius = 0.15 * self.orbit_radius

# Planetary data
PLANETARY_DATA = {
    'Mercury': {'mass': 3.3011e23, 'orbit_radius': 0.387 * AU, 'lagrange_points': None},
    'Venus': {'mass': 4.8675e24, 'orbit_radius': 0.723 * AU, 'lagrange_points': None},
    'Earth': {'mass': 5.97237e24, 'orbit_radius': 1.0 * AU, 'lagrange_points': None},
    'Mars': {'mass': 6.4171e23, 'orbit_radius': 1.52 * AU, 'lagrange_points': None},
    'Jupiter': {'mass': 1.8982e27, 'orbit_radius': 5.20 * AU, 'lagrange_points': None},
    'Saturn': {'mass': 5.6834e26, 'orbit_radius': 9.54 * AU, 'lagrange_points': None},
    'Uranus': {'mass': 8.6810e25, 'orbit_radius': 19.19 * AU, 'lagrange_points': None},
    'Neptune': {'mass': 1.02413e26, 'orbit_radius': 30.07 * AU, 'lagrange_points': None}
}

class Planet:
    def __init__(self, name: str, mass: float, orbit_radius: float):
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mass = torch.tensor(mass, device=self.device)
        self.orbit_radius = torch.tensor(orbit_radius, device=self.device)
        self.position = torch.zeros(2, device=self.device)
        self.velocity = torch.zeros(2, device=self.device)
        self.radius = torch.tensor(0.0, device=self.device)  # Yarıçap özelliği eklendi
        self.positions = []
        self.l1_positions = []
        self.l2_positions = []

        # Lagrange noktalarını hesapla
        self.lagrange = LagrangePoints(mass, orbit_radius)
        
    def gravitational_acceleration(self, r: torch.Tensor) -> torch.Tensor:
        r_mag = torch.norm(r)
        if r_mag < 1e-10:
            return torch.zeros_like(r)
        return -G * SUN_MASS * r / (r_mag**3)

    def update_position_rk45(self, dt_and_state=None):
        def derivatives(t_and_state: tuple) -> torch.Tensor:
            t, state = t_and_state
            r, v = state[:2], state[2:]
            a = self.gravitational_acceleration(r)
            return torch.cat([v, a])

        # Eğer dt_and_state tuple olarak geldiyse, ayrıştır
        if isinstance(dt_and_state, tuple):
            dt, (pos, vel) = dt_and_state
            y = torch.cat([pos, vel])
        else:
            dt = dt_and_state  # Normal dt değeri
            y = torch.cat([self.position, self.velocity])
        
        # Zaman noktalarını önceden hesapla
        t0 = torch.zeros(1, device=self.device)
        t1 = torch.tensor([dt/4], device=self.device)
        t2 = torch.tensor([3*dt/8], device=self.device)
        t3 = torch.tensor([12*dt/13], device=self.device)
        t4 = torch.tensor([dt], device=self.device)
        t5 = torch.tensor([dt/2], device=self.device)
        
        # RK45 katsayıları
        k1 = dt * derivatives((t0, y))
        k2 = dt * derivatives((t1, y + k1/4))
        k3 = dt * derivatives((t2, y + 3*k1/32 + 9*k2/32))
        k4 = dt * derivatives((t3, y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197))
        k5 = dt * derivatives((t4, y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
        k6 = dt * derivatives((t5, y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
        
       
        y4 = y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        y5 = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
        
        # Hata tahmini
        error = torch.norm(y5 - y4)
        
        # Optimal adım boyutu hesaplama
        tolerance = 1e-6
        optimal_dt = dt * (tolerance / (2 * error))**0.2
        
        # Eğer hata çok büyükse adım boyutunu küçült ve tekrar dene
        if error > tolerance:
            return self.update_position_rk45((optimal_dt, (y5[:2], y5[2:])))
            
        # Pozisyon ve hızı güncelle
        self.position = y5[:2]
        self.velocity = y5[2:]

        # Lagrange noktalarını güncelle
        angle = torch.atan2(self.position[1], self.position[0])
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)]
        ], device=self.device)
        
        l1_pos = torch.matmul(rotation_matrix, self.lagrange.l1)
        l2_pos = torch.matmul(rotation_matrix, self.lagrange.l2)
        
        self.l1_positions.append(l1_pos)
        self.l2_positions.append(l2_pos)
        self.positions.append(self.position.clone())

    def reset(self):
        """Gezegeni başlangıç konumuna döndür"""
        if self.name in self.orbital_parameters:
            params = self.orbital_parameters[self.name]
            
            # Başlangıç açısını tensor olarak bir kez oluştur
            initial_angle = params['initial_angle'].to(self.device)
            
            # Trigonometrik hesaplamaları bir kez yap
            cos_angle = torch.cos(initial_angle)
            sin_angle = torch.sin(initial_angle)
            
            # Yarıçapı hesapla
            r = params['semi_major_axis'] * (1 - params['eccentricity']**2) / \
                (1 + params['eccentricity'] * cos_angle)
            
            # Pozisyon vektörünü oluştur
            self.position = torch.stack([
                r * cos_angle,
                r * sin_angle
            ]).to(self.device)
            
            # Hız büyüklüğünü hesapla
            velocity_magnitude = torch.sqrt(G * SUN_MASS * 
                                         (2/r - 1/params['semi_major_axis']))
            
            # Hız vektörünü oluştur
            self.velocity = torch.stack([
                -velocity_magnitude * sin_angle,
                velocity_magnitude * cos_angle
            ]).to(self.device)

class Spacecraft:
    def __init__(self, mass: float):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Temel özellikler
        self.mass = torch.tensor(mass, device=device, dtype=torch.float32)  # Mass özelliği eklendi
        self.dry_mass = torch.tensor(mass, device=device, dtype=torch.float32)
        self.velocity = torch.zeros(2, device=device, dtype=torch.float32)  # [0.0, 1500.0] yerine zeros kullanıldı
        self.acceleration = torch.zeros(2, device=device, dtype=torch.float32)
         # Başlangıç hızını ayrıca ayarlayın
        self.velocity[1] = 1500.0  # Y ekseni hızını 1500 olarak ayarla
        # Durum vektörleri - pozisyon, hız, ivme
        self.position = torch.zeros(2, device=self.device)
        self.velocity = torch.zeros(2, device=self.device)
        self.acceleration = torch.zeros(2, device=self.device)
        
        # Geçmiş kayıtları
        self.thrust_history: List[float] = []
        self.speed_history: List[float] = []
        self.acceleration_history: List[float] = []
        self.position_history: List[torch.Tensor] = []
        self.fuel_consumption_history: List[float] = []
        self.orbit_parameters_history: List[Dict[str, float]] = []
        
        # Motor sistemleri
        self.main_engine_thrust = torch.tensor(50000.0, device=device, dtype=torch.float32)  # Newton
        self.backup_engines = 3
        self.current_engine = 0
        self.engine_efficiency = torch.tensor(0.95, device=device, dtype=torch.float32)
        self.engine_temperature = torch.tensor(300.0, device=device, dtype=torch.float32)  # Kelvin
        self.min_engine_temp = torch.tensor(200.0, device=device, dtype=torch.float32)  # Kelvin
        self.max_engine_temp = torch.tensor(1500.0, device=device, dtype=torch.float32)  # Kelvin
        self.optimal_temp_range = (800.0, 1200.0)  # Optimal çalışma sıcaklığı aralığı
        self.heating_rate = torch.tensor(2.0, device=device, dtype=torch.float32)  # Kelvin/s
        self.cooling_rate = torch.tensor(1.0, device=device, dtype=torch.float32)  # Kelvin/s
        
        # Yakıt sistemi
        self.fuel_consumption_rate = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.has_fuel_leak = False
        self.fuel_leak_rate = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.fuel_system_health = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # Yapısal bütünlük
        self.hull_integrity = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.hull_damage = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.critical_damage_threshold = torch.tensor(0.8, device=device, dtype=torch.float32)
        
        # Onarım sistemleri
        self.repair_in_progress = False
        self.repair_time_remaining = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.repair_efficiency = torch.tensor(0.8, device=device, dtype=torch.float32)
        self.auto_repair_systems = True
        
        # Navigasyon sistemleri
        self.navigation_accuracy = torch.tensor(0.99, device=device, dtype=torch.float32)
        self.guidance_system_health = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.trajectory_error = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # Güç sistemleri
        self.power_level = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.solar_panel_efficiency = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.battery_charge = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # Isı yönetimi
        self.thermal_load = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.cooling_system_efficiency = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.radiator_performance = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # Hayat destek sistemleri
        self.life_support_health = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.oxygen_level = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.pressure_integrity = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # İletişim sistemleri
        self.communication_strength = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.signal_quality = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.data_transmission_rate = torch.tensor(1.0, device=device, dtype=torch.float32)
        
        # Performans metrikleri
        self.delta_v_remaining = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.mission_time = torch.tensor(0.0, device=device, dtype=torch.float32)
        self.total_distance_traveled = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # Sistem durumu izleme
        self.system_health = {
            'propulsion': torch.tensor(1.0, device=device, dtype=torch.float32),
            'navigation': torch.tensor(1.0, device=device, dtype=torch.float32),
            'power': torch.tensor(1.0, device=device, dtype=torch.float32),
            'thermal': torch.tensor(1.0, device=device, dtype=torch.float32),
            'life_support': torch.tensor(1.0, device=device, dtype=torch.float32),
            'communication': torch.tensor(1.0, device=device, dtype=torch.float32),
            'structural': torch.tensor(1.0, device=device, dtype=torch.float32)
        }
        
        # Acil durum sistemleri
        self.emergency_systems_active = False
        self.emergency_power_reserve = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.emergency_protocols = {
            'engine_shutdown': False,
            'power_conservation': False,
            'damage_control': False,
            'life_support_backup': False
        }
        
        # Sensör sistemleri
        self.sensors = {
            'radiation': torch.tensor(0.0, device=device, dtype=torch.float32),
            'temperature': torch.tensor(300.0, device=device, dtype=torch.float32),
            'pressure': torch.tensor(1.0, device=device, dtype=torch.float32),
            'acceleration': torch.tensor(0.0, device=device, dtype=torch.float32),
            'magnetic_field': torch.tensor(0.0, device=device, dtype=torch.float32)
        }
        
        # Performans sınırları
        self.limits = {
            'max_acceleration': torch.tensor(MAX_ACCELERATION, device=device, dtype=torch.float32),
            'max_speed': torch.tensor(MAX_SPACECRAFT_SPEED, device=device, dtype=torch.float32),
            'max_temp': torch.tensor(2000.0, device=device, dtype=torch.float32),
            'min_temp': torch.tensor(100.0, device=device, dtype=torch.float32),
            'max_radiation': torch.tensor(1000.0, device=device, dtype=torch.float32),
            'max_pressure': torch.tensor(2.0, device=device, dtype=torch.float32)
        }

    def apply_action(self, action):
       """Uzay aracına eylemi uygula"""
       try:
           # Eylemi tensor'a çevir
           if isinstance(action, np.ndarray):
               action = torch.from_numpy(action).float().to(self.device)
               
           # İtki yönü ve büyüklüğünü ayır
           thrust_direction = action[:2]  # x, y yönü
           thrust_magnitude = torch.abs(action[2])  # İtki büyüklüğü
           
           # İtki yönünü normalize et
           thrust_direction = thrust_direction / (torch.norm(thrust_direction) + 1e-8)
           
           # İtki kuvvetini hesapla (Newton)
           MAX_THRUST = 1000.0  # Newton
           thrust_force = thrust_magnitude * MAX_THRUST * thrust_direction
           
           # Kuvveti uygula
           self.acceleration = thrust_force / self.mass
           
       except Exception as e:
           print(f"Action uygulama hatası: {e}")

    def update(self, dt: float):
        """Uzay aracının durumunu güncelle"""
        try:
           # Tensor boyutlarını kontrol et
            assert self.position.size() == (2,), f"Position size error: {self.position.size()}"
            assert self.velocity.size() == (2,), f"Velocity size error: {self.velocity.size()}"
            assert self.acceleration.size() == (2,), f"Acceleration size error: {self.acceleration.size()}"
            
            # dt'yi tensor'a çevir
            dt = torch.tensor(dt, device=self.device)
              
           # Vektör boyutlarını kontrol et ve düzelt
            if self.velocity.shape != (2,):
               self.velocity = self.velocity[:2]
            if self.acceleration.shape != (2,):
               self.acceleration = self.acceleration[:2]
            
            # Pozisyonu güncelle
            self.position = self.position + self.velocity * dt + 0.5 * self.acceleration * dt * dt
           
           # Hızı güncelle
            self.velocity = self.velocity + self.acceleration * dt
           
           # Hız sınırlaması
            speed = torch.norm(self.velocity)
            if speed > MAX_SPACECRAFT_SPEED:
               self.velocity *= (MAX_SPACECRAFT_SPEED / speed)
            
            # Yakıt sızıntısı varsa yakıt kaybını hesapla
            if self.has_fuel_leak:
                fuel_loss = self.fuel_leak_rate * dt
                self.fuel_mass = max(0.0, self.fuel_mass - fuel_loss)
                self.mass = self.dry_mass + self.fuel_mass
            
            # Motor sıcaklığını güncelle
            if torch.norm(self.acceleration) > 0:
                # Motor çalışıyorsa ısınma
                self.engine_temperature = torch.clamp(
                    self.engine_temperature + self.heating_rate * dt,
                    min=self.min_engine_temp,
                    max=self.max_engine_temp
                )
                
                # Aşırı ısınma durumunda motor hasarı
                if self.engine_temperature > self.max_engine_temp * 0.9:
                    damage_factor = (self.engine_temperature - self.max_engine_temp * 0.9) / (self.max_engine_temp * 0.1)
                    self.hull_damage += damage_factor * 0.01 * dt
            else:
                # Motor çalışmıyorsa soğuma
                self.engine_temperature = torch.clamp(
                    self.engine_temperature - self.cooling_rate * dt,
                    min=self.min_engine_temp,
                    max=self.max_engine_temp
                )
            
            # Onarım işlemlerini güncelle
            if self.repair_in_progress:
                self.repair_time_remaining = max(0.0, self.repair_time_remaining - dt)
                if self.repair_time_remaining == 0:
                    self.hull_damage *= (1 - self.repair_efficiency)
                    self.repair_in_progress = False
            
            # Sistem durumlarını güncelle
            self.update_emergency_systems(dt)
            
            # Geçmiş kayıtlarını güncelle
            self._update_history()
            
        except Exception as e:
           print(f"Update hatası: {e}")
           print(f"Position size: {self.position.size()}")
           print(f"Velocity size: {self.velocity.size()}")
           print(f"Acceleration size: {self.acceleration.size()}")
           raise e

    def _update_history(self):
        """Geçmiş kayıtlarını güncelle"""
        self.thrust_history.append(float(torch.norm(self.acceleration)))
        self.speed_history.append(float(torch.norm(self.velocity)))
        self.acceleration_history.append(float(torch.norm(self.acceleration)))
        self.position_history.append(self.position.clone())
        self.fuel_consumption_history.append(float(self.fuel_mass))

    def get_system_status(self) -> Dict[str, float]:
        """Sistemlerin genel durumunu döndürür."""
        return {
            'hull_integrity': float(self.hull_integrity),
            'fuel_remaining': float(self.fuel_mass),
            'power_level': float(self.power_level),
            'engine_health': float(self.system_health['propulsion']),
            'navigation_accuracy': float(self.navigation_accuracy),
            'life_support': float(self.life_support_health),
            'communication': float(self.communication_strength),
            'thermal_load': float(self.thermal_load)
        }

    def get_engine_efficiency(self) -> float:
        """Motor verimliliğini hesapla"""
        if self.engine_temperature < self.min_engine_temp:
            return 0.0
        elif self.engine_temperature > self.max_engine_temp:
            return 0.0
        elif self.optimal_temp_range[0] <= self.engine_temperature <= self.optimal_temp_range[1]:
            return 1.0
        else:
            # Optimal aralık dışındaysa azalan verimlilik
            if self.engine_temperature < self.optimal_temp_range[0]:
                t_diff = (self.engine_temperature - self.min_engine_temp) / (self.optimal_temp_range[0] - self.min_engine_temp)
            else:
                t_diff = (self.max_engine_temp - self.engine_temperature) / (self.max_engine_temp - self.optimal_temp_range[1])
            return max(0.0, min(1.0, t_diff))
        
    def apply_thrust(self, action: torch.Tensor) -> None:
        """Uzay aracına itki uygula"""
        # Action vektörünü parçalara ayır
        thrust_direction = action[:2]  # İlk 2 eleman yön vektörü
        thrust_magnitude = action[2]   # 3. eleman itki büyüklüğü
        correction = action[3]         # 4. eleman düzeltme manevrası
        
        # Yön vektörünü normalize et
        thrust_direction = thrust_direction / (torch.norm(thrust_direction) + 1e-8)
        
        # İtki kuvvetini hesapla (Newton)
        max_thrust = 5000.0  # Maximum thrust in Newtons
        thrust_force = thrust_magnitude * max_thrust * thrust_direction
        
        # İvmeyi hesapla (F = ma)
        self.acceleration = thrust_force / self.mass
        
        # Düzeltme manevrasını uygula
        correction_force = torch.tensor([
            -self.velocity[1],
            self.velocity[0]
        ], device=self.device) * correction
        
        # Toplam ivmeyi güncelle
        self.acceleration += correction_force / self.mass
        
        # Motor sıcaklığını güncelle
        self.engine_temperature += self.heating_rate * thrust_magnitude
        self.engine_temperature = max(self.min_engine_temp, 
                                        min(self.max_engine_temp, 
                                            self.engine_temperature))
        
        # Yakıt tüketimi
        fuel_consumption = thrust_magnitude * 0.1  # kg/s
        self.fuel_mass = max(0.0, self.fuel_mass - fuel_consumption)    
     
    def apply_force(self, force: torch.Tensor, thrust_direction: torch.Tensor = None, 
                    thrust_magnitude: float = 0.0, correction: float = 0.0):
        """Uzay aracına kuvvet uygula"""
        try:
            total_force = force.clone()
            
            # İtki kuvveti varsa ekle
            if thrust_direction is not None and thrust_magnitude > 0:
                # Yön vektörnü normalize et
                thrust_direction = thrust_direction / (torch.norm(thrust_direction) + 1e-8)
                
                # İtki kuvvetini hesapla
                max_thrust = 5000.0  # Maximum thrust in Newtons
                thrust_force = thrust_magnitude * max_thrust * thrust_direction
                total_force += thrust_force
                
                # Düzeltme manevrasını uygula
                if correction != 0:
                    correction_force = torch.tensor([
                        -self.velocity[1],
                        self.velocity[0]
                    ], device=self.device) * correction
                    total_force += correction_force
                
                # Motor sıcaklığını güncelle
                self.engine_temperature += self.heating_rate * thrust_magnitude
                self.engine_temperature = torch.clamp(
                    self.engine_temperature,
                    min=self.min_engine_temp,
                    max=self.max_engine_temp
                )
                
                # Yakıt tüketimi
                fuel_consumption = thrust_magnitude * 0.1  # kg/s
                self.fuel_mass = max(0.0, self.fuel_mass - fuel_consumption)
                self.mass = self.dry_mass + self.fuel_mass
            
            # İvmeyi hesapla (F = ma)
            self.acceleration = total_force / self.mass
            
        except Exception as e:
            print(f"Kuvvet uygulama hatası: {e}")
            traceback.print_exc()

    def apply_gravity_forces(self, planets: List[Planet]):
        total_force = torch.zeros(2, device=self.device)
        for planet in planets:
            r = planet.position - self.position
            r_mag = torch.norm(r)
            force = G * planet.mass * self.mass * r / (r_mag ** 3)
            total_force += force
        self.acceleration = total_force / self.mass
    
    def update(self, dt: float):
       """Uzay aracının durumunu güncelle"""
       try:
           dt = torch.tensor(dt, device=self.device)
           
           # Acceleration tensor'unu düzelt (2,4) -> (2,)
           if self.acceleration.dim() > 1:
               self.acceleration = self.acceleration[:, 0]  # İlk sütunu al
           
           # Boyut kontrolü
           self.acceleration = self.acceleration[:2]  # Sadece x,y bileşenlerini al
           self.velocity = self.velocity[:2]
           self.position = self.position[:2]
           
           # Güncelleme
           delta_pos = (self.velocity * dt) + (0.5 * self.acceleration * dt * dt)
           self.position = self.position + delta_pos
           self.velocity = self.velocity + (self.acceleration * dt)
           
       except Exception as e:
           print(f"Update hatası detayı: {e}")
           print(f"Position boyutu: {self.position.size()}")
           print(f"Velocity boyutu: {self.velocity.size()}")
           print(f"Acceleration boyutu: {self.acceleration.size()}")
           print(f"dt değeri: {dt}")
    
    def get_emergency_status(self) -> Dict[str, bool]:
        """Acil durum sistemlerinin durumunu döndürür."""
        return {
            'emergency_active': self.emergency_systems_active,
            'engine_shutdown': self.emergency_protocols['engine_shutdown'],
            'power_conservation': self.emergency_protocols['power_conservation'],
            'damage_control': self.emergency_protocols['damage_control'],
            'life_support_backup': self.emergency_protocols['life_support_backup']
        }

    def get_sensor_readings(self) -> Dict[str, float]:
        """Sensör okumalarını döndürür."""
        return {key: float(value) for key, value in self.sensors.items()}

    def get_performance_metrics(self) -> Dict[str, float]:
        """Performans metriklerini döndürür."""
        return {
            'delta_v_remaining': float(self.delta_v_remaining),
            'mission_time': float(self.mission_time),
            'distance_traveled': float(self.total_distance_traveled),
            'current_speed': float(torch.norm(self.velocity)),
            'current_acceleration': float(torch.norm(self.acceleration))
        }
    
    def update_emergency_systems(self, time_step: float):
        # Print mesajları için zaman kontrolü
        should_print = not hasattr(self, '_last_warning_time') or \
                    time_step - self._last_warning_time > 300  # 5 dakikada bir uyarı
        
        # Geliştirilmiş motor sıcaklık yönetimi
        if torch.norm(self.acceleration) > 0:
            heat_generation = time_step * 0.05 * torch.norm(self.acceleration)
            cooling_power = time_step * 0.1 * self.cooling_system_efficiency
            
            if self.engine_temperature > 1000:
                self.cooling_system_efficiency *= 0.999
            
            net_temperature_change = heat_generation - cooling_power
            old_temp = float(self.engine_temperature)
            
            # Sıcaklık değişimini sınırla ve uygula
            self.engine_temperature = torch.clamp(
                self.engine_temperature + net_temperature_change,
                min=self.min_engine_temp,
                max=self.max_engine_temp
            )
            
            if self.engine_temperature > self.max_engine_temp * 0.9:
                self.system_health['propulsion'] *= 0.995
                
                # Acil durum soğutması
                if self.engine_temperature > self.max_engine_temp * 0.95:
                    emergency_cooling = (self.engine_temperature - self.max_engine_temp * 0.95) * 0.1
                    self.engine_temperature = torch.clamp(
                        self.engine_temperature - emergency_cooling,
                        min=self.min_engine_temp,
                        max=self.max_engine_temp
                    )
                    self.power_level *= 0.99
                    
                # Sadece önemli değişikliklerde ve belirli aralıklarla print
                if should_print and (
                    self.system_health['propulsion'] < 0.5 or 
                    abs(float(self.engine_temperature - old_temp)) > 500 
                ):
                    print(f"\nSistem Durumu:")
                    print(f"- Motor Sıcaklığı: {float(self.engine_temperature):.1f}K")
                    print(f"- Motor Sağlığı: {float(self.system_health['propulsion'])*100:.1f}%")
                    print(f"- Soğutma Verimi: {float(self.cooling_system_efficiency)*100:.1f}%")
                    self._last_warning_time = time_step
        else:
            # Motor çalışmıyorken soğuma
            cooling_rate = time_step * 0.2 * self.cooling_system_efficiency
            self.engine_temperature = torch.clamp(
                self.engine_temperature - cooling_rate,
                min=self.min_engine_temp,
                max=self.max_engine_temp
            )

        # Hayat destek sistemleri güncellemesi
        if self.power_level < 0.5:
            self.life_support_health *= 0.99
            self.oxygen_level *=0.99
            if should_print and self.life_support_health < 0.8:
                print("\nKritik Sistem Uyarısı:")
                print(f"- Hayat Destek Sistemi: {float(self.life_support_health)*100:.1f}%")
                print(f"- Oksijen Seviyesi: {float(self.oxygen_level)*100:.1f}%")
                self._last_warning_time = time_step

        # Acil durum protokolleri kontrolü
        critical_systems = {
            'propulsion': self.system_health['propulsion'] < 0.5,
            'power': self.power_level < 0.3,
            'life_support': self.life_support_health < 0.7,
            'structural': self.hull_integrity < 0.6
        }

        if any(critical_systems.values()) and should_print:
            print("\nACİL DURUM RAPORU:")
            for system, is_critical in critical_systems.items():
                if is_critical:
                    print(f"- {system.upper()} sistemi kritik seviyede!")
            self._last_warning_time = time_step

        # Motor güvenlik protokolü
        if self.engine_temperature > self.max_engine_temp * 0.95 and should_print:
            if not hasattr(self, '_engine_shutdown_announced'):
                print("\nACİL DURUM: Motor güvenlik protokolü devrede!")
                self._engine_shutdown_announced = True
            self.emergency_protocols['engine_shutdown'] = True
            self.acceleration *= 0.1
    
    def leapfrog_step(self, dt: float, force: torch.Tensor):
        """
        Leapfrog integrasyon yöntemi ile uzay aracının hareketini günceller.
        """
        # İvmeyi güncelle
        self.acceleration = force / self.mass
        
        # Yarım adım hız güncellemesi
        half_velocity = self.velocity + 0.5 * dt * self.acceleration
        
        # Pozisyon güncellemesi
        self.position += dt * half_velocity
        
        # İvmeyi tekrar hesapla (yeni pozisyonda)
        self.acceleration = force / self.mass
        
        # Hızın kalan yarısını güncelle
        self.velocity = half_velocity + 0.5 * dt * self.acceleration
        
        # Hız sınırlaması
        current_acceleration = torch.norm(self.acceleration)
        max_safe_acceleration = self.limits['max_acceleration'] * self.system_health['propulsion']
        
        if current_acceleration > max_safe_acceleration:
            old_acceleration = float(current_acceleration)
            self.acceleration *= (max_safe_acceleration / current_acceleration)
            
            # Sadece büyük değişikliklerde ve aralıklı olarak print
            if (not hasattr(self, '_last_acceleration_warning') or 
                dt - self._last_acceleration_warning > 300) and \
                abs(old_acceleration - float(max_safe_acceleration)) > 10:
                
                print(f"\nİvme Kontrolü: {old_acceleration:.2f} -> {float(max_safe_acceleration):.2f} m/s²")
                self._last_acceleration_warning = dt
            
            # İvme sınırlaması
            current_acceleration = torch.norm(self.acceleration)
            if current_acceleration > self.limits['max_acceleration']:
                self.acceleration *= (self.limits['max_acceleration'] / current_acceleration)
            
            # Geçmiş kayıtlarını güncelle
            self.position_history.append(self.position.clone())
            self.speed_history.append(float(torch.norm(self.velocity)))
            self.acceleration_history.append(float(current_acceleration))
            
            # Yörünge parametrelerini güncelle
            self.update_orbit_parameters()

    def update_orbit_parameters(self):
        """Yörünge parametrelerini günceller."""
        try:
            r = self.position
            v = self.velocity
            
            # 2D uzayda çalışıyoruz, z bileşeni 0 olan 3D vektörler oluştur
            r_3d = torch.cat([r, torch.tensor([0.0], device=device, dtype=torch.float32)])
            v_3d = torch.cat([v, torch.tensor([0.0], device=device, dtype=torch.float32)])
            
            # Spesifik açısal momentum (3D cross product için)
            h = torch.cross(r_3d.unsqueeze(0), v_3d.unsqueeze(0))[0]
            
            # Yarı-major eksen
            r_mag = torch.norm(r)
            v_mag = torch.norm(v)
            a = -G * SUN_MASS / (v_mag**2 - 2 * G * SUN_MASS / r_mag)
            
            # Eksantrisite vektörü
            e_vec = torch.cross(v_3d.unsqueeze(0), h.unsqueeze(0))[0] / (G * SUN_MASS) - r_3d / r_mag
            e = torch.norm(e_vec)
            
            # İnklinasyon (2D'de her zaman 0)
            i = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            # Yörünge parametrelerini kaydet
            orbit_params = {
                'semi_major_axis': float(a),
                'eccentricity': float(e),
                'inclination': float(i),
                'angular_momentum': float(torch.norm(h))
            }
            
            self.orbit_parameters_history.append(orbit_params)
            
        except Exception as e:
            print(f"Yörünge parametreleri hesaplanırken hata: {e}")
            # Hata durumunda varsayılan değerler
            self.orbit_parameters_history.append({
                'semi_major_axis': 0.0,
                'eccentricity': 0.0,
                'inclination': 0.0,
                'angular_momentum': 0.0
            })

    def update_mass(self, fuel_consumed: float):
        """Yakıt tüketimini güncelleyerek kütle değişimini hesaplar."""
        fuel_consumed = torch.tensor(fuel_consumed, device=device, dtype=torch.float32)
        self.fuel_mass = torch.maximum(
            torch.tensor(0.0, device=device, dtype=torch.float32),
            self.fuel_mass - fuel_consumed
        )
        self.mass = self.dry_mass + self.fuel_mass
        self.fuel_consumption_history.append(float(fuel_consumed))

    def handle_motor_failure(self) -> bool:
        """Motor arızasını yönetir."""
        if self.backup_engines > 0:
            self.backup_engines -= 1
            self.current_engine += 1
            print(f"Motor arızası! Yedek motor #{self.current_engine} devrede.")
            print(f"Kalan yedek motor: {self.backup_engines}")
            return True
        else:
            print("Kritik: Tüm motorlar arızalı!")
            return False

    def handle_fuel_leak(self) -> bool:
        """Yakıt sızıntısını yönetir."""
        if not self.has_fuel_leak:
            self.has_fuel_leak = True
            self.fuel_leak_rate = torch.tensor(
                random.uniform(0.0001, 0.001),
                device=device,
                dtype=torch.float32
            )
            print(f"Yakıt sızıntısı tespit edildi! Sızıntı oranı: {self.fuel_leak_rate*100:.4f}%/s")
            return True
        else:
            print("Kritik: İkinci yakıt sızıntısı! Sistem başarısız!")
            return False

    def handle_microasteroid_collision(self) -> bool:
        """Mikroasteroit çarpışmasını yönetir."""
        damage = torch.tensor(
            random.uniform(0.01, 0.05),
            device=device,
            dtype=torch.float32
        )
        self.hull_damage += damage
        print(f"Mikro asteroit çarpışması! Hasar: {damage*100:.1f}%, Toplam hasar: {self.hull_damage*100:.1f}%")
        
        if self.hull_damage >= self.critical_damage_threshold:
            print("Kritik: Gövde bütünlğü kritik seviyede!")
            return False
        
        if not self.repair_in_progress and self.auto_repair_systems:
            self.repair_in_progress = True
            self.repair_time_remaining = torch.tensor(3600.0, device=device, dtype=torch.float32)
        
        return True

    def calculate_orbital_elements(self, params):
        initial_angle = params['true_anomaly']
        
        # Açıyı tensor olarak bir kez oluştur
        angle_tensor = torch.tensor(initial_angle, device=self.device)
        cos_angle = torch.cos(angle_tensor)
        sin_angle = torch.sin(angle_tensor)
        
        # Yörünge yarıçapı
        r = params['semi_major_axis'] * \
            (1 - params['eccentricity']**2) / \
            (1 + params['eccentricity'] * cos_angle)
        
        # Pozisyon vektörü
        position = torch.tensor([
            r * cos_angle,
            r * sin_angle
        ], device=self.device)
        
        # Hız büyüklüğü
        velocity_magnitude = torch.sqrt(G * SUN_MASS * (2/r - 1/params['semi_major_axis']))
        
        # Hız vektörü
        velocity = torch.tensor([
            -velocity_magnitude * sin_angle,
            velocity_magnitude * cos_angle
        ], device=self.device)
        
        return position, velocity

class TransformerMemory(nn.Module):
    def __init__(self, state_size = 41, action_size = 4, memory_size = 1000):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.d_model = 256  # Transformer model boyutu
        self.nhead = 8     # Multi-head attention başlık sayısı
        self.num_layers = 4  # Transformer katman sayısı
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Bellek matrisi [state, action, reward, next_state]
        self.memory = torch.zeros(
            memory_size, 
            2*state_size + action_size + 1,  # [state + next_state + action + reward]
            device=self.device
        )
        
        # Pozisyon kodlaması
        self.pos_encoder = nn.Parameter(
            self._create_position_encoding(), 
            requires_grad=False
        )
        
        # Giriş projeksiyonu
        self.input_proj = nn.Sequential(
            nn.Linear(2*state_size + action_size + 1, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        # Transformer kodlayıcı
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Çıkış projeksiyonu
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, action_size)
        )
        
        # Bellek indeksi
        self.current_idx = 0
        
        # Önem ağırlıkları
        self.importance_net = nn.Sequential(
            nn.Linear(2*state_size + action_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)
        
    def _create_position_encoding(self):
        """Sinüzoidal pozisyon kodlaması oluştur"""
        position = torch.arange(self.memory_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.memory_size, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def add_experience(self, state, action, reward, next_state):
        """Yeni deneyimi belleğe ekle"""
        # Deneyimi birleştir
        experience = torch.cat([
            state.flatten(), 
            action.flatten(),
            reward.view(-1),
            next_state.flatten()
        ])
        
        # Önem değerini hesapla
        importance = self.importance_net(experience.unsqueeze(0)).item()
        
        # Eğer bellek doluysa ve yeni deneyim önemliyse
        if self.current_idx >= self.memory_size:
            # En düşük önemli deneyimi bul
            importances = torch.tensor([
                self.importance_net(self.memory[i].unsqueeze(0)).item()
                for i in range(self.memory_size)
            ])
            min_idx = torch.argmin(importances)
            
            # Yeni deneyim daha önemliyse değiştir
            if importance > importances[min_idx]:
                self.memory[min_idx] = experience
        else:
            # Bellek dolmamışsa direkt ekle
            self.memory[self.current_idx] = experience
            self.current_idx = (self.current_idx + 1) % self.memory_size
    
    def query_memory(self, current_state: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Mevcut duruma en benzer k deneyimi getir"""
        with torch.no_grad():
            # Mevcut durumu kodla
            state_encoding = self.input_proj(current_state.unsqueeze(0))
            
            # Belleği kodla
            memory_encoded = self.input_proj(self.memory) + self.pos_encoder
            
            # Attention skorlarını hesapla
            attention_scores = torch.matmul(
                state_encoding, memory_encoded.transpose(-2, -1)
            ) / math.sqrt(self.d_model)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # En benzer k deneyimi seç
            _, indices = torch.topk(attention_weights.squeeze(), k)
            selected_memories = self.memory[indices]
            
            # Transformer ile işle
            encoded = self.transformer(
                self.input_proj(selected_memories) + self.pos_encoder[:k]
            )
            
            # Çıktıyı hesapla
            output = self.output_proj(encoded.mean(dim=0))
            
            return output

    def get_memory_influence(self, state: torch.Tensor) -> torch.Tensor:
        """Belleğin mevcut duruma etkisini hesapla"""
        with torch.no_grad():
            # En benzer deneyimleri al
            memory_output = self.query_memory(state)
            
            # Bellek etkisini hesapla (0-1 arası)
            memory_influence = torch.sigmoid(
                torch.norm(memory_output) / math.sqrt(self.action_size)  # Kapanış parantezi eklendi
            )
            
            return memory_influence

class SACNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size  # Yeni eklenen satır
        self.action_size = action_size  # Yeni eklenen satır
        
        # Actor ağı eklendi
        self.actor = nn.Sequential(
            nn.Linear(state_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        ).to(self.device)
        
        # Genişletilmiş Critic ağları
        self.q1_network = nn.Sequential(
            nn.Linear(state_size + action_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        self.q2_network = nn.Sequential(
            nn.Linear(state_size + action_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # Genişletilmiş Value ağı
        self.value_network = nn.Sequential(
            nn.Linear(state_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # Target value ağı
        self.target_value_network = copy.deepcopy(self.value_network)
        
        # Mean ve log_std başlıkları
        self.mean_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        ).to(self.device)
        
        self.log_std_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        ).to(self.device)
        
        # Optimizerlar
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + 
            list(self.mean_head.parameters()) + 
            list(self.log_std_head.parameters()), 
            lr=3e-4
        )
        self.q1_optimizer = optim.Adam(self.q1_network.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2_network.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=3e-4)
        
        # Entropy ayarı için alpha parametresi
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_size

        # [Mevcut PINN katmanları aynı kalacak...]

        # Physics encoder eklendi
        self.physics_encoder = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        ).to(self.device)
        
        # Physics forward metodu için conservation layer eklendi
        self.conservation_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 4)  # [energy, momentum_x, momentum_y, momentum_z]
        ).to(self.device)
        
        # Relativistic layer eklendi
        self.relativistic_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)  # gamma factor
        ).to(self.device)
        
        # Scaler eklendi
        self.scaler = torch.amp.GradScaler('cuda')

        # Ağırlık başlatma
        self.apply(self._init_weights)

        # Adaptif öğrenme oranı için parametreler
        self.initial_lr = 3e-4
        self.min_lr = 1e-5
        self.lr_decay = 0.995
        
        # Sıcaklık parametresi için adaptif ayarlama
        self.target_entropy = -action_size  # Hedef entropi değeri
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Gradient clipping değeri
        self.max_grad_norm = 1.0
        
        # Optimizerları güncelle - adaptif öğrenme oranı ile
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + 
            list(self.mean_head.parameters()) + 
            list(self.log_std_head.parameters()),
            lr=self.initial_lr
        )
        
        self.q1_optimizer = optim.Adam(self.q1_network.parameters(), lr=self.initial_lr)
        self.q2_optimizer = optim.Adam(self.q2_network.parameters(), lr=self.initial_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.initial_lr)
        
        # Gelişmiş deneyim replay buffer
        self.buffer_size = 1_000_000
        self.batch_size = 256
        self.replay_buffer = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': [],
            'priority': []  # Öncelikli deneyim replay için
        }
        self.per_alpha = 0.6  # Öncelik üssü
        self.per_beta = 0.4   # Önemli örnekleme düzeltme faktörü
        self.per_epsilon = 1e-6  # Küçük pozitif değer

    def _init_weights(self, module):
        """Ağırlıkları Xavier/Glorot başlatması ile başlat"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward_critic(self, state, action):
        """Genişletilmiş critic ileri yayılımı"""
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        return q1, q2

    def forward_value(self, state):
        """Genişletilmiş value ileri yayılımı"""
        return self.value_network(state)

    def forward(self, state):
        """Genişletilmiş ileri yayılım"""
        try:
            # Giriş boyutunu kontrol et ve düzelt
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Batch boyutu ve özellik sayısını ayır
            batch_size = state.size(0)
            
            # State'i doğru boyuta getir
            if state.size(-1) != self.state_size:
                # Eksik özellikleri sıfırlarla doldur
                padded_state = torch.zeros(batch_size, self.state_size, device=self.device)
                padded_state[:, :state.size(-1)] = state
                state = padded_state
            
            # Actor özellikleri
            with torch.amp.autocast('cuda'):
                actor_features = self.actor(state)
                actor_features = torch.clamp(actor_features, -10.0, 10.0)
                
                # Mean ve log_std hesaplama
                mean = self.mean_head(actor_features)
                log_std = self.log_std_head(actor_features)
                
                # Sınırlama ve stabilizasyon
                mean = torch.clamp(mean, -1.0, 1.0)
                log_std = torch.clamp(log_std, -5.0, 2.0)
                
                # Fizik çıktısı
                with torch.no_grad():
                    physics_output = self.physics_forward(state)
                    physics_output = torch.clamp(physics_output, -1.0, 1.0)
                
                # Adaptif ağırlıklandırma
                alpha = torch.sigmoid(self.log_alpha)
                combined_mean = alpha * mean + (1 - alpha) * physics_output
                
                # Son sınırlama
                combined_mean = torch.clamp(combined_mean, -1.0, 1.0)
                
                return combined_mean, log_std
                
        except Exception as e:
            print(f"Forward pass hatası: {e}")
            print(f"Giriş boyutu: {state.shape}")
            print(f"Beklenen boyut: {self.state_size}")
            return (torch.zeros((state.size(0), self.action_size), device=self.device),
                    torch.full((state.size(0), self.action_size), -5.0, device=self.device))
                
        except Exception as e:
            print(f"Forward pass hatası: {e}")
            return (torch.zeros((state.size(0), self.action_size), device=self.device),
                    torch.full((state.size(0), self.action_size), -5.0, device=self.device))

    def sample_action(self, state):
        """Geliştirilmiş aksiyon örnekleme"""
        try:
            mean, log_std = self.forward(state)
            
            # NaN kontrolü
            mean = torch.nan_to_num(mean, nan=0.0)
            log_std = torch.nan_to_num(log_std, nan=-20.0)
            
            std = torch.exp(log_std) + 1e-6  # Sayısal stabilite için epsilon ekle
            
            # Normal dağılım örneklemesi
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            
            # Tanh sınırlama
            action = torch.tanh(x_t)
            
            # Log olasılık hesaplama
            log_prob = normal.log_prob(x_t)
            log_prob = torch.nan_to_num(log_prob, nan=0.0)  # NaN kontrolü
            
            # Tanh düzeltmesi
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            
            # Son NaN kontrolü
            action = torch.nan_to_num(action, nan=0.0)
            log_prob = torch.nan_to_num(log_prob, nan=0.0)
            
            return action, log_prob
            
        except Exception as e:
            print(f"Sample action hatası: {e}")
            return (torch.zeros_like(mean), 
                    torch.zeros((mean.size(0), 1), device=self.device))

    def physics_forward(self, state):
        """Fizik tabanlı forward pass"""
        physics_features = self.physics_encoder(state)
        conservation_pred = self.conservation_layer(physics_features)
        relativistic_pred = self.relativistic_layer(physics_features)
        
        return conservation_pred[:, :self.action_size]  # Sadece eylem boyutuna kadar al

    def update_learning_rate(self, episode: int):
        """Öğrenme oranını adaptif olarak güncelle"""
        for optimizer in [self.actor_optimizer, self.q1_optimizer, 
                         self.q2_optimizer, self.value_optimizer]:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(self.min_lr, current_lr * self.lr_decay)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def add_experience(self, state, action, reward, next_state, done):
        """Gelişmiş deneyim ekleme"""
        # TD hatası ile öncelik hesapla
        with torch.no_grad():
            current_q1, current_q2 = self.forward_critic(state, action)
            next_value = self.forward_value(next_state)
            target_q = reward + (1 - done) * self.gamma * next_value
            td_error = abs(target_q - torch.min(current_q1, current_q2)).item()
        
        # Öncelikli deneyim replay için öncelik hesapla
        priority = (td_error + self.per_epsilon) ** self.per_alpha
        
        # Buffer'a ekle
        if len(self.replay_buffer['state']) >= self.buffer_size:
            # En düşük öncelikli deneyimi çıkar
            min_priority_idx = np.argmin(self.replay_buffer['priority'])
            for key in self.replay_buffer:
                self.replay_buffer[key].pop(min_priority_idx)
        
        # Yeni deneyimi ekle
        self.replay_buffer['state'].append(state)
        self.replay_buffer['action'].append(action)
        self.replay_buffer['reward'].append(reward)
        self.replay_buffer['next_state'].append(next_state)
        self.replay_buffer['done'].append(done)
        self.replay_buffer['priority'].append(priority)

    def sample_batch(self):
        """Öncelikli örnekleme ile batch seç"""
        buffer_size = len(self.replay_buffer['state'])
        if buffer_size < self.batch_size:
            return None
        
        # Önceliklere göre örnekleme olasılıkları
        probs = np.array(self.replay_buffer['priority'])
        probs = probs / np.sum(probs)
        
        # Öncelikli örnekleme
        indices = np.random.choice(
            buffer_size, 
            self.batch_size, 
            p=probs,
            replace=False
        )
        
        # Importance sampling ağırlıkları
        weights = (buffer_size * probs[indices]) ** (-self.per_beta)
        weights = weights / weights.max()
        
        # Batch'i hazırla
        batch = {
            'state': torch.stack([self.replay_buffer['state'][i] for i in indices]),
            'action': torch.stack([self.replay_buffer['action'][i] for i in indices]),
            'reward': torch.tensor([self.replay_buffer['reward'][i] for i in indices], device=self.device),
            'next_state': torch.stack([self.replay_buffer['next_state'][i] for i in indices]),
            'done': torch.tensor([self.replay_buffer['done'][i] for i in indices], device=self.device),
            'weights': torch.FloatTensor(weights).to(self.device),
            'indices': indices
        }
        
        return batch

    def update(self, batch):
        """Geliştirilmiş güncelleme fonksiyonu"""
        # Batch verilerini al
        states = batch['state']
        actions = batch['action'] 
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']
        weights = batch['weights']
        
        # Value loss
        with torch.amp.autocast('cuda'):
            current_value = self.forward_value(states)
            next_value = self.forward_value(next_states)
            next_value = next_value * (1 - dones)
            target_value = rewards + self.gamma * next_value
            value_loss = (weights * F.mse_loss(current_value, target_value.detach(), reduction='none')).mean()
        
        # Optimize value network
        self.value_optimizer.zero_grad()
        self.scaler.scale(value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        self.scaler.step(self.value_optimizer)
        
        # Q losses
        with torch.amp.autocast('cuda'):
            current_q1, current_q2 = self.forward_critic(states, actions)
            q_loss = (weights * (F.mse_loss(current_q1, target_value.detach(), reduction='none') + 
                               F.mse_loss(current_q2, target_value.detach(), reduction='none'))).mean()
        
        # Optimize Q networks
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.scaler.scale(q_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q1_network.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.q2_network.parameters(), self.max_grad_norm)
        self.scaler.step(self.q1_optimizer)
        self.scaler.step(self.q2_optimizer)
        
        # Policy loss
        with torch.amp.autocast('cuda'):
            new_actions, log_probs = self.sample_action(states)
            alpha = self.log_alpha.exp()
            q1_new = self.forward_critic(states, new_actions, self.q1_network)
            q2_new = self.forward_critic(states, new_actions, self.q2_network)
            q_new = torch.min(q1_new, q2_new)
            policy_loss = (weights * (alpha * log_probs - q_new)).mean()
        
        # Optimize policy network
        self.actor_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.scaler.step(self.actor_optimizer)
        
        # Alpha loss - adaptif sıcaklık parametresi
        with torch.amp.autocast('cuda'):
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optimizer)
        
        # Scaler güncelle
        self.scaler.update()
        
        # Öncelikleri güncelle
        td_errors = abs(target_value - torch.min(current_q1, current_q2)).detach().cpu().numpy()
        new_priorities = (td_errors + self.per_epsilon) ** self.per_alpha
        for idx, priority in zip(batch['indices'], new_priorities):
            self.replay_buffer['priority'][idx] = priority
        
        return {
            'value_loss': value_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }

class SACAgent:
    def __init__(self, state_size: int, action_size: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        
        # SAC ağları
        self.sac = SACNetwork(state_size, action_size).to(self.device)
        
        # Hiperparametreler
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 1024
        self.memory_size = 100000
        self.memory = []

        self.target_entropy = -action_size  # Hedef entropi değeri
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
       
       # Boyut düzeltmeleri için value ve critic ağlarının çıktı boyutlarını ayarla
        self.sac.value_network[-1] = nn.Linear(256, 1)  # Son katmanı 1 boyutlu çıktı verecek şekilde değiştir
        self.sac.q1_network[-1] = nn.Linear(256, 1)     # Q ağları için de aynı şekilde
        self.sac.q2_network[-1] = nn.Linear(256, 1)
        self.q1_network = self.sac.q1_network
        self.q2_network = self.sac.q2_network

        # Optimizer'lara referanslar ekle
        self.value_optimizer = self.sac.value_optimizer
        self.q1_optimizer = self.sac.q1_optimizer
        self.q2_optimizer = self.sac.q2_optimizer
        self.actor_optimizer = self.sac.actor_optimizer
        self.alpha_optimizer = self.sac.alpha_optimizer

         # Scaler'ı da ekle
        self.scaler = torch.amp.GradScaler('cuda')
        
    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Duruma göre eylem seç"""
        with torch.no_grad():  # Gradient hesaplamayı devre dışı bırak
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Batch boyutu kontrolü
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            # Belleği temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            action, _ = self.sac.sample_action(state)
            return action.squeeze(0).cpu().numpy()

    def forward_value(self, states):
       """Value network için forward pass"""
       return self.sac.forward_value(states)
   
   # Yeni metot ekle
    def forward_critic(self, states, actions, network=None):
       """Critic network için forward pass"""
       if network is None:
           return self.sac.forward_critic(states, actions)
       return network(torch.cat([states, actions], dim=-1))
    def sample_action(self, states):
       """Durumlar için eylem örnekle"""
       return self.sac.sample_action(states)

    def replay(self):
        """Deneyim replay ile öğrenme"""
        if len(self.memory) < self.batch_size:
            return {'value_loss': 0, 'q_loss': 0, 'policy_loss': 0, 'alpha_loss': 0}
        
        # Batch örnekleme
        batch = random.sample(self.memory, self.batch_size)
        
        # Batch verilerini tensor'lara dönüştür ve GPU'ya taşı
        states = torch.stack([torch.as_tensor(x[0], device=self.device) for x in batch])
        actions = torch.stack([torch.as_tensor(x[1], device=self.device) for x in batch])
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([torch.as_tensor(x[3], device=self.device) for x in batch])
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)
        
        # Modeli GPU'ya taşı
        self.sac = self.sac.to(self.device)
            # Value loss
        with torch.amp.autocast('cuda'):
            current_value = self.forward_value(states).view(-1, 1)
            next_value = self.forward_value(next_states).view(-1, 1)
            next_value = next_value * (1 - dones).view(-1, 1)
            target_value = rewards.view(-1, 1) + self.gamma * next_value
            value_loss = F.mse_loss(current_value, target_value.detach())
            
            # Q losses
            current_q1, current_q2 = self.forward_critic(states, actions)
            q_loss = F.mse_loss(current_q1, target_value.detach()) + \
                    F.mse_loss(current_q2, target_value.detach())
            
            # Policy loss
            new_actions, log_probs = self.sample_action(states)
            alpha = self.log_alpha.exp()
            q1_new = self.forward_critic(states, new_actions, self.q1_network).view(-1, 1)
            q2_new = self.forward_critic(states, new_actions, self.q2_network).view(-1, 1)
            q_new = torch.min(q1_new, q2_new)
            policy_loss = (alpha * log_probs - q_new).mean()
            
            # Alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            # Value network optimization
        self.value_optimizer.zero_grad()
        self.scaler.scale(value_loss).backward(retain_graph=True)
        # Infinity kontrolü ekle
        self.scaler.unscale_(self.value_optimizer)
        torch.nn.utils.clip_grad_norm_(self.sac.value_network.parameters(), 1.0)
        self.scaler.step(self.value_optimizer)
            # Q networks optimization
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.scaler.scale(q_loss).backward(retain_graph=True)
        # Infinity kontrolü ekle
        self.scaler.unscale_(self.q1_optimizer)
        self.scaler.unscale_(self.q2_optimizer)
        torch.nn.utils.clip_grad_norm_(self.sac.q1_network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.sac.q2_network.parameters(), 1.0)
        self.scaler.step(self.q1_optimizer)
        self.scaler.step(self.q2_optimizer)
            # Policy network optimization
        self.actor_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward(retain_graph=True)
        # Infinity kontrolü ekle
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.sac.actor.parameters(), 1.0)
        self.scaler.step(self.actor_optimizer)
            # Alpha optimization
        self.alpha_optimizer.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        # Infinity kontrolü ekle
        self.scaler.unscale_(self.alpha_optimizer)
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
        self.scaler.step(self.alpha_optimizer)
            # Scaler güncelle
        self.scaler.update()
            # Soft update target network
        for target_param, param in zip(self.sac.target_value_network.parameters(), 
                                        self.sac.value_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            return {
            'value_loss': value_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item()
        }

    def save_model(self, path):
        torch.save({
            'sac_state_dict': self.sac.state_dict(),
            'value_optimizer_state_dict': self.sac.value_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.sac.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.sac.q2_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.sac.policy_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.sac.alpha_optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.sac.load_state_dict(checkpoint['sac_state_dict'])
        self.sac.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.sac.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.sac.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.sac.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.sac.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

class LongTermSACNetwork(SACNetwork):
    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        
        # Uzun vadeli bellek mekanizması
        self.memory = TransformerMemory(state_size, action_size)
        self.actor = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        ).to(self.device)
         # Gelişmiş deneyim replay buffer
        self.buffer_size = 1_000_000
        self.batch_size = 1024
         # Mean ve log_std başlıkları
        self.mean_head = nn.Linear(256, action_size).to(self.device)
        self.log_std_head = nn.Linear(256, action_size).to(self.device)
        # Bellek entegrasyonu için ek katmanlar
        self.memory_integration = nn.ModuleDict({
            'state_gate': nn.Sequential(
                nn.Linear(state_size * 2, state_size),
                nn.Sigmoid()
            ),
            'action_gate': nn.Sequential(
                nn.Linear(action_size * 2, action_size),
                nn.Sigmoid()
            )
        })
    def forward_actor(self, state):
        """Actor network için forward pass"""
        features = self.actor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Çıktıları sınırla
        mean = torch.clamp(mean, -1.0, 1.0)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        
        return mean, log_std
    
    def forward_with_memory(self, state):
        # Normal forward pass
        mean, log_std = self.forward_actor(state)
        
        # Bellekten benzer deneyimleri sorgula
        memory_output = self.memory.query_memory(state)
        memory_state = memory_output[:, :self.state_size]  # Batch boyutunu koru
        memory_action = memory_output[:, self.state_size:self.state_size + self.action_size]
        
        # Bellek entegrasyonu için kapı değerlerini hesapla
        state_gate = self.memory_integration['state_gate'](
            torch.cat([state, memory_state], dim=-1)
        )
        action_gate = self.memory_integration['action_gate'](
            torch.cat([mean, memory_action], dim=-1)
        )
        
        # Gated entegrasyon - Tek bir geçişte hesapla
        integrated_mean = action_gate * mean + (1 - action_gate) * memory_action
        
        # Bellek durumunu güncelle
        integrated_state = state_gate * state + (1 - state_gate) * memory_state
        
        # Bellek durumunu güncelle ve kullan
        integrated_state = state_gate * state + (1 - state_gate) * memory_state
        integrated_mean = self.mean_head(integrated_state)  # Entegre durumu mean hesabında kullan
        
        return integrated_mean, log_std
    
    def update_memory(self, state, action, reward):
        """Belleği güncelle"""
        # Boyut kontrolü ekle
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
            
        self.memory.add_experience(state, action, reward)

class LongTermSACAgent(SACAgent):
    def __init__(self, state_size: int, action_size: int):
        super().__init__(state_size, action_size)
        self.fitness_score = 0.0
        self.sac = LongTermSACNetwork(state_size, action_size).to(self.device)
    def get_fitness(self):
        """Ajanın fitness skorunu döndür"""
        return self.fitness_score
    
    def update_fitness(self, reward):
        """Fitness skorunu güncelle"""
        self.fitness_score += reward

    def act(self, state):
        """Tek bir durum için eylem seç"""
        try:
            # State'i tensor'a çevir
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            if not isinstance(state, torch.Tensor):
                raise ValueError(f"Unexpected state type: {type(state)}")
                
            with torch.no_grad():
                # Boyut kontrolü ve düzeltmesi
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                
                # Forward pass ve eylem örnekleme
                action, _ = self.sac.forward_with_memory(state)
                
                # NaN kontrolü
                if torch.isnan(action).any():
                    print("Uyarı: NaN eylem üretildi")
                    action = torch.zeros_like(action)
                
                return action.cpu().numpy()[0]
                
        except Exception as e:
            print(f"Act metodu hatası: {e}")
            return np.random.uniform(-1, 1, self.action_size)
    
    def remember(self, state, action, reward, next_state, done):
        super().remember(state, action, reward, next_state, done)
        # Belleği güncelle
        self.sac.update_memory(
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor([reward]).to(self.device)
        )

class GeneticSACAgent:
    def __init__(self, state_size = 11, action_size = 4, population_size = 48):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.batch_size = 256
        self.gamma = 0.99  # İndirim faktörü
        # Genetik algoritma parametreleri
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = max(2, population_size // 8)
        self.tournament_size = 3
        self.memory = []
        self.memory_size = 100000
        # Popülasyon oluştur
        self.population = [SACAgent(state_size, action_size) for _ in range(population_size)]
        self.fitness_scores = torch.zeros(population_size, device=self.device)
        self.generation = 0
        self.actor = None  # Önce mevcut actor'ü temizle
        if hasattr(self, 'actor'):
            del self.actor
        torch.cuda.empty_cache()  # GPU belleğini temizle
        self.actor = nn.Sequential(
            nn.Linear(state_size, 512),  # 11 -> 256
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_size)  # 256 -> 4
        ).to(self.device)
        
        # Ağırlıkları sıfırla
        for layer in self.actor.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        self.value_network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        self.q1_network = nn.Sequential(
            nn.Linear(state_size + action_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        self.q2_network = nn.Sequential(
            nn.Linear(state_size + action_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        # Q optimizerları ekle
        self.q1_optimizer = optim.Adam(self.q1_network.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2_network.parameters(), lr=3e-4)
        self.mean_head = nn.Linear(256, action_size).to(self.device)
        self.log_std_head = nn.Linear(256, action_size).to(self.device)  
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=3e-4)
        self.log_std_min = -20
        self.log_std_max = 2      
        # Target entropy için negatif action_size
        self.target_entropy = -action_size
        # Target entropy ve log_alpha ekle
        self.target_entropy = -action_size  # Hedef entropi değeri
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.mean_head = nn.Linear(256, action_size).to(self.device)
        print(f"Actor first layer weight shape: {self.actor[0].weight.shape}")  # Bu (256, 11)
    def sample_action(self, state):
        """Duruma göre eylem örnekle"""
        try:
            mean, log_std = self.forward(state)
            
            # log_std'yi sınırla
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            # Normal dağılımdan örnekle
            std = log_std.exp()
            normal = torch.randn_like(mean)
            
            # Reparametrization trick
            action = mean + std * normal
            
            # tanh ile sınırla
            action = torch.tanh(action)
            
            # Log probability hesapla
            log_prob = (-0.5 * ((normal ** 2) + 2 * log_std + np.log(2 * np.pi))).sum(dim=-1)
            
            # tanh düzeltmesi
            log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            
            return action, log_prob.unsqueeze(-1)
            
        except Exception as e:
            print(f"Sample action hatası: {e}")
            return (torch.zeros_like(mean), 
                   torch.zeros((state.size(0), 1), device=self.device))
          
    def select_parent(self) -> SACAgent:
        tournament = random.sample(list(enumerate(self.population)), self.tournament_size)
        tournament_fitness = [self.fitness_scores[idx] for idx, _ in tournament]
        winner_idx = tournament[torch.argmax(torch.tensor(tournament_fitness))][0]
        return self.population[winner_idx]
    
    def crossover(self, parent1: SACAgent, parent2: SACAgent) -> SACAgent:
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)
            
        child = SACAgent(self.state_size, self.action_size)
        
        for (name1, param1), (name2, param2) in zip(
            parent1.sac.named_parameters(),
            parent2.sac.named_parameters()
        ):
            if random.random() < 0.5:
                dict(child.sac.named_parameters())[name1].data.copy_(param1.data)
            else:
                dict(child.sac.named_parameters())[name1].data.copy_(param2.data)
                
        return child
    
    def mutate(self, agent: SACAgent):
        for param in agent.sac.parameters():
            if random.random() < self.mutation_rate:
                noise = torch.randn_like(param) * 0.1
                param.data += noise
                param.data.clamp_(-1, 1)
    
    def evolve(self):
        """Popülasyonu evrimleştir"""
        try:
            # Popülasyonu ödüllere göre sırala
            sorted_population = sorted(self.population, 
                                    key=lambda x: x.get_fitness(),
                                    reverse=True)
            
            # En iyi bireyleri sakla
            elite_size = max(1, int(self.population_size * self.elite_ratio))
            new_population = sorted_population[:elite_size]
            
            # Kalan popülasyonu oluştur
            while len(new_population) < self.population_size:
                # Turnuva seçimi
                parent1 = self._tournament_select(sorted_population)
                parent2 = self._tournament_select(sorted_population)
                
                try:
                    # Çaprazlama
                    child = self.crossover(parent1, parent2)
                    
                    # Mutasyon
                    if random.random() < self.mutation_rate:
                        self.mutate(child)
                    
                    new_population.append(child)
                except Exception as e:
                    print(f"Birey üretme hatası: {e}")
                    # Hata durumunda elite bireylerden birini kopyala
                    new_population.append(deepcopy(sorted_population[0]))
            
            self.population = new_population
            
        except Exception as e:
            print(f"Evrim hatası: {e}")
            traceback.print_exc()
    
    def step_batch(self, actions):
        """Batch halinde adım at"""
        try:
            next_states = []
            rewards = []
            dones = []
            infos = []
            
            # Her uzay aracı için paralel simülasyon
            for i in range(self.num_envs):
                # NumPy dizisini tensor'a dönüştür
                if isinstance(actions, np.ndarray):
                    action = torch.from_numpy(actions[i]).float().to(self.device)
                else:
                    action = actions[i]
                    
                # Her bir ajan için adım at
                with torch.no_grad():
                    if action.dim() == 1:
                        action = action.unsqueeze(0)
                        
                    # Eylemi uygula ve sonuçları al
                    next_state = self.get_state()
                    reward = self._calculate_reward(i)
                    done = self._check_episode_end(i)
                    info = {}
                    
                    next_states.append(next_state[i])
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
            
            # Sonuçları tensor'a dönüştür
            next_states = torch.stack(next_states).to(self.device)
            rewards = torch.tensor(rewards, device=self.device)
            dones = torch.tensor(dones, device=self.device)
            
            return next_states, rewards, dones, infos
                
        except Exception as e:
            print(f"Step batch hatası: {e}")
            return (torch.zeros((self.num_envs, self.state_size), device=self.device),
                    torch.zeros(self.num_envs, device=self.device),
                    torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
                    [{} for _ in range(self.num_envs)])
        
    def update_fitness(self, agent_idx: int, reward: float):
        self.fitness_scores[agent_idx] += reward

    def remember_batch(self, states, actions, rewards, next_states, dones):
       """Batch halinde deneyimleri belleğe ekle"""
       try:
           # Batch boyutunu al
           batch_size = len(states)
           
           # Her bir deneyimi belleğe ekle
           for i in range(batch_size):
               if len(self.memory) >= self.memory_size:
                   self.memory.pop(0)
                   
               self.memory.append((
                   states[i],
                   actions[i],
                   rewards[i],
                   next_states[i],
                   dones[i]
               ))
               
       except Exception as e:
           print(f"Remember batch hatası: {e}")

    def act_batch(self, states):
        """Batch halinde eylem seç"""
        try:
            if isinstance(states, np.ndarray):
                states = torch.FloatTensor(states).to(self.device)
            
            # Batch boyutu kontrolü
            if states.dim() == 1:
                states = states.unsqueeze(0)
            
            with torch.no_grad():
                actions = self.actor(states)
                return actions.cpu().numpy()
                
        except Exception as e:
            print(f"Act batch hatası: {e}")
            return np.zeros((states.shape[0], self.action_size))
        
    def act(self, state):
        """Tek bir durum için eylem seç"""
        try:
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            with torch.no_grad():
                action = self.actor(state)
                return action.squeeze(0).cpu().numpy()
                
        except Exception as e:
            print(f"Act metodu hatası: {e}")
            return np.zeros(self.action_size)
        
    def forward_value(self, state):
        """Value network için forward pass"""
        return self.value_network(state)
    
    def forward(self, state):
        """Genişletilmiş ileri yayılım"""
        try:
            # Giriş boyutunu kontrol et ve düzelt
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)

            # Batch boyutu kontrolü
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Actor özellikleri
            actor_features = self.actor[:-1](state)  # Son katmanı hariç tut
            # Mean ve log_std başlıklarını uygula
            mean = self.mean_head(actor_features)
            log_std = self.log_std_head(actor_features)
            
            # Çıktıları sınırla ve boyutları kontrol et
            mean = torch.clamp(mean, -1.0, 1.0)
            log_std = torch.clamp(log_std, -5.0, 2.0)
            
            # Boyut kontrolü ekle
            if mean.dim() == 1:
                mean = mean.unsqueeze(0)
            if log_std.dim() == 1:
                log_std = log_std.unsqueeze(0)
            
            return mean, log_std
             
        except Exception as e:
            print(f"Forward pass hatası: {e}")
            return (torch.zeros((1, self.action_size), device=self.device),
                    torch.full((1, self.action_size), -5.0, device=self.device))
        
    def forward_critic(self, states, actions, network=None):
        """Critic networks için forward pass"""
        try:
            # Boyut kontrolü ve düzeltmesi
            if states.dim() == 1:
                states = states.unsqueeze(0)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
                
            # Batch boyutlarını eşitle
            if states.size(0) != actions.size(0):
                if states.size(0) == 1:
                    states = states.expand(actions.size(0), -1)
                elif actions.size(0) == 1:
                    actions = actions.expand(states.size(0), -1)
            
            # State ve action'ları birleştir
            x = torch.cat([states, actions], dim=-1)
            
            if network is None:
                # Her iki Q değerini hesapla
                q1 = self.q1_network(x)
                q2 = self.q2_network(x)
                return q1, q2
            else:
                # Belirtilen network için Q değerini hesapla
                return network(x)
                
        except Exception as e:
            print(f"Forward critic hatası: {e}")
            if network is None:
                return (torch.zeros((states.size(0), 1), device=self.device), 
                    torch.zeros((states.size(0), 1), device=self.device))
            else:
                return torch.zeros((states.size(0), 1), device=self.device)
    
    def replay(self):
        """Deneyim replay ile öğrenme"""
        if len(self.memory) < self.batch_size:
            return {
                'value_loss': 0.0,
                'q_loss': 0.0,
                'policy_loss': 0.0,
                'alpha_loss': 0.0
            }
        
        try:
            # Batch örnekleme
            batch = random.sample(self.memory, self.batch_size)
            
            # Batch verilerini tensor'lara dönüştür ve boyutları düzelt
            states = []
            for x in batch:
                state = x[0]
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if state.dim() == 1:
                    state = state.unsqueeze(0)  # [11] -> [1, 11]
                elif state.size(0) == 48:
                    state = state[0].unsqueeze(0)  # [48, 11] -> [1, 11]
                states.append(state.to(self.device))
            states = torch.cat(states, dim=0)  # stack yerine cat kullan
            
            # Benzer şekilde actions için de boyut düzeltmesi yap
            actions = []
            for x in batch:
                action = x[1]
                if isinstance(action, np.ndarray):
                    action = torch.FloatTensor(action)
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                elif action.size(0) == 48:
                    action = action[0].unsqueeze(0)
                actions.append(action.to(self.device))
            actions = torch.cat(actions, dim=0)
            
            # Diğer verileri tensor'a dönüştür
            rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
            
            # Next states için de aynı boyut düzeltmelerini yap
            next_states = []
            for x in batch:
                next_state = x[3]
                if isinstance(next_state, np.ndarray):
                    next_state = torch.FloatTensor(next_state)
                if next_state.dim() == 1:
                    next_state = next_state.unsqueeze(0)
                elif next_state.size(0) == 48:
                    next_state = next_state[0].unsqueeze(0)
                next_states.append(next_state.to(self.device))
            next_states = torch.cat(next_states, dim=0)
            
            dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)
            
            # Geri kalan kod aynı...
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                current_value = self.forward_value(states).view(-1, 1)  # Boyutu [batch_size, 1] yap
                next_value = self.forward_value(next_states).view(-1, 1)
                next_value = next_value * (1 - dones).view(-1, 1)
                target_value = rewards.view(-1, 1) + self.gamma * next_value
                value_loss = F.mse_loss(current_value, target_value.detach())
                    
                # Q losses
                current_q1, current_q2 = self.forward_critic(states, actions)
                q_loss = F.mse_loss(current_q1, target_value.detach()) + \
                        F.mse_loss(current_q2, target_value.detach())
                
                # Policy loss
                new_actions, log_probs = self.sample_action(states)
                alpha = self.log_alpha.exp()
                q1_new = self.forward_critic(states, new_actions, self.q1_network)
                q2_new = self.forward_critic(states, new_actions, self.q2_network)
                q_new = torch.min(q1_new, q2_new)
                policy_loss = (alpha * log_probs - q_new).mean()
                 # Alpha hesaplama
                alpha = self.log_alpha.exp()
                
                # Policy loss
                new_actions, log_probs = self.sample_action(states)
                q1_new = self.forward_critic(states, new_actions, self.q1_network)
                q2_new = self.forward_critic(states, new_actions, self.q2_network)
                q_new = torch.min(q1_new, q2_new)
                policy_loss = (alpha * log_probs - q_new).mean()
                
                # Alpha loss
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                # Alpha optimization
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward(retain_graph=True)
                self.alpha_optimizer.step()
            
            return {
                'value_loss': float(value_loss.detach().cpu().item()),
                'q_loss': float(q_loss.detach().cpu().item()),
                'policy_loss': float(policy_loss.detach().cpu().item()),
                'alpha_loss': float(alpha_loss.detach().cpu().item())
            }
            
        except Exception as e:
            print(f"Replay hatası detayı: {e}")
            traceback.print_exc()
            return {
                'value_loss': 0.0,
                'q_loss': 0.0,
                'policy_loss': 0.0,
                'alpha_loss': 0.0
            }

    def save_population(self, path: str):
        """Tüm popülasyonu kaydet"""
        try:
            population_states = []
            for i, agent in enumerate(self.population):
                if hasattr(agent, 'sac'):
                    population_states.append({
                        'agent_idx': i,
                        'sac_state_dict': agent.sac.state_dict(),
                        'fitness': float(self.fitness_scores[i])
                    })
            torch.save(population_states, path)
        except Exception as e:
            print(f"Popülasyon kaydetme hatası: {e}")

class LongTermGeneticSACAgent(GeneticSACAgent):
    def __init__(self, state_size: int, action_size: int, population_size: int = 48):
        super().__init__(state_size, action_size, population_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.elite_ratio = 0.2
        self.population = [LongTermSACAgent(state_size, action_size) 
                for _ in range(population_size)]
        # Popülasyonu LongTermSACAgent'lardan oluştur
        self.population = [
            LongTermSACAgent(state_size, action_size) for _ in range(population_size)
        ]
        
        # Diğer parametreler aynı
        self.elite_size = 12
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.fitness_scores = torch.zeros(population_size, device=self.device)
        
    def _tournament_select(self, population):
        """Turnuva seçimi yap"""
        try:
            # Turnuva için rastgele bireyler seç
            tournament_size = 3  # Turnuva büyüklüğü
            tournament = random.sample(population, tournament_size)
            
            # En yüksek fitness değerine sahip bireyi seç
            winner = max(tournament, key=lambda x: x.get_fitness())
            return winner
            
        except Exception as e:
            print(f"Turnuva seçimi hatası: {e}")
            # Hata durumunda popülasyondan rastgele bir birey döndür
            return random.choice(population)
        
    def crossover(self, parent1: LongTermSACAgent, parent2: LongTermSACAgent) -> LongTermSACAgent:
        """Bellek mekanizmasını da içeren çaprazlama"""
        child = super().crossover(parent1, parent2)
        
        # Bellek entegrasyonu için ek parametreleri çaprazla
        for (name1, param1), (name2, param2) in zip(
            parent1.sac.memory_integration.named_parameters(),
            parent2.sac.memory_integration.named_parameters()
        ):
            if random.random() < self.crossover_rate:
                mask = torch.rand_like(param1) < 0.5
                new_param = torch.where(mask, param1, param2)
                dict(child.sac.memory_integration.named_parameters())[name1].data.copy_(new_param)
        
        return child

class SpaceTravelEnv:
    def __init__(self, destination_planet_name: str, num_envs: int = 1, num_workers: int = 16):
        """
        Uzay yolculuğu simülasyon ortamını başlat
        
        Args:
            destination_planet_name: Hedef gezegenin adı
            num_envs: Paralel simülasyon sayısı (varsayılan: 1)
        """
        # Önce cihaz ve fizik sabitlerini tanımla
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        self.time_step = 86400.0
        self.num_envs = num_envs
        self.num_workers = num_workers 
        # Zaman adımını tanımla - 1 gün (saniye cinsinden)
        self.dt = self.time_step 
        
        # Fizik sabitleri - En başta tanımlanmalı
        self.G = torch.tensor(G, device=self.device, dtype=torch.float32)  # Gravitasyon sabiti
        self.sun_mass = torch.tensor(SUN_MASS, device=self.device, dtype=torch.float32)  # Güneş kütlesi
        
        # Orbital parametreleri tanımla
        self.orbital_parameters = {
            'Mercury': {
                'semi_major_axis': torch.tensor(0.387 * AU, device=self.device),
                'eccentricity': torch.tensor(0.206, device=self.device),
                'orbital_period': torch.tensor(87.97 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(0.0, device=self.device)
            },
            'Venus': {
                'semi_major_axis': torch.tensor(0.723 * AU, device=self.device),
                'eccentricity': torch.tensor(0.007, device=self.device),
                'orbital_period': torch.tensor(224.7 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(torch.pi / 4, device=self.device)
            },
            'Earth': {
                'semi_major_axis': torch.tensor(1.0 * AU, device=self.device),
                'eccentricity': torch.tensor(0.017, device=self.device),
                'orbital_period': torch.tensor(365.25 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(torch.pi / 2, device=self.device)
            },
            'Mars': {
                'semi_major_axis': torch.tensor(1.524 * AU, device=self.device),
                'eccentricity': torch.tensor(0.093, device=self.device),
                'orbital_period': torch.tensor(687.0 *24 *3600, device=self.device),
                'initial_angle': torch.tensor(3 * torch.pi / 4, device=self.device)
            },
            'Jupiter': {
                'semi_major_axis': torch.tensor(5.203 * AU, device=self.device),
                'eccentricity': torch.tensor(0.048, device=self.device),
                'orbital_period': torch.tensor(4331.0 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(torch.pi, device=self.device)
            },
            'Saturn': {
                'semi_major_axis': torch.tensor(9.537 * AU, device=self.device),
                'eccentricity': torch.tensor(0.054, device=self.device),
                'orbital_period': torch.tensor(10747.0 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(5 * torch.pi / 4, device=self.device)
            },
            'Uranus': {
                'semi_major_axis': torch.tensor(19.191 * AU, device=self.device),
                'eccentricity': torch.tensor(0.047, device=self.device),
                'orbital_period': torch.tensor(30589.0 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(3 * torch.pi / 2, device=self.device)
            },
            'Neptune': {
                'semi_major_axis': torch.tensor(30.069 * AU, device=self.device),
                'eccentricity': torch.tensor(0.009, device=self.device),
                'orbital_period': torch.tensor(59800.0 * 24 * 3600, device=self.device),
                'initial_angle': torch.tensor(7 * torch.pi / 4, device=self.device)
            }
        }
        
        # Fizik motoru
        self.physics_engine = RelativisticSpacePhysics(device=self.device)
        
        # Gezegenleri oluştur
        self.planets = self._create_planets()
        self.destination = self.planets[destination_planet_name]
        
        # Uzay araçlarını oluştur
        self.spacecraft = [Spacecraft(1000.0) for _ in range(num_envs)]
        
        # Başlangıç durumunu ayarla
        self._initialize_spacecraft()
        
        # Durum ve eylem boyutlarını hesapla
        self._state_size = self._calculate_state_size()
        self.state_size = self._state_size  # Dışarıdan erişim için
        self.action_size = 4  # [thrust_x, thrust_y, thrust_magnitude, correction]
        
        # Simülasyon zamanı ve durum
        self.time = torch.zeros(num_envs, device=self.device)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # Geçmiş kayıtları
        self.fuel_history = []
        self.distance_history = []
        self.speed_history = []
        self.time_history = []
        self.acceleration_history = []
        
        # İlk mesafeyi kaydet
        self.initial_distance = torch.norm(
            self.spacecraft[0].position - self.destination.position
        )
        self.previous_distance = torch.full(
            (num_envs,), float('inf'), device=self.device
        )
        
        # Adım sayacını ekle
        self.current_step = torch.zeros(num_envs, device=self.device)
        self.max_steps = 1000  # Maksimum adım sayısı
        # Popülasyon zelliğini ekle
        self.population = self.spacecraft  # spacecraft listesini population olarak kullan
        
    def _calculate_state_size(self) -> int:
        """Durum vektörünün boyutunu hesapla"""
        # Basitleştirilmiş state boyutu
        spacecraft_state = {
            'position': 2,          # x, y pozisyonu
            'velocity': 2,          # x, y hızı
            'fuel': 1,             # yakıt miktarı
        }
        
        # Hedef gezegen durumu
        target_state = {
            'position': 2,          # x, y pozisyonu
            'velocity': 2,          # x, y hızı
            'distance': 1,          # uzay aracına olan mesafe
        }
        
        # Genel metrikler
        general_metrics = {
            'mission_time': 1,      # görev süresi
        }
        
        # Toplam boyutu hesapla
        total_size = (sum(spacecraft_state.values()) + 
                        sum(target_state.values()) + 
                        sum(general_metrics.values()))
        
        return total_size
    
    def get_state(self):
        """Güncel durumu döndür - optimize edilmiş versiyon"""
        try:
            # Uzay araçlarının durumlarını tek seferde hesapla
            positions = torch.stack([s.position for s in self.spacecraft])
            velocities = torch.stack([s.velocity for s in self.spacecraft])
            fuel_masses = torch.tensor([[s.fuel_mass] for s in self.spacecraft], device=self.device)
            
            # Uzay aracı durumlarını birleştir
            spacecraft_states = torch.cat([positions, velocities, fuel_masses], dim=1)
            
            # Hedef gezegen durumunu tekrarla
            target_pos_vel = torch.cat([
                self.destination.position,
                self.destination.velocity
            ]).repeat(self.num_envs, 1)
            
            # Zamanı ekle
            time_tensor = self.time.view(-1, 1)
            
            # Tüm durumları birleştir
            states = torch.cat([
                spacecraft_states,  # [batch_size, 5]
                target_pos_vel,    # [batch_size, 4]
                time_tensor        # [batch_size, 1]
            ], dim=1)
            
            return states
            
        except Exception as e:
            print(f"State hesaplama hatası: {e}")
            return torch.zeros((self.num_envs, self._state_size), device=self.device)

    def step_batch(self, actions):
        """Batch halinde adım at"""
        try:
            next_states = []
            rewards = []
            dones = []
            infos = []
            batch_size = len(actions)
            batch_per_worker = batch_size // self.num_workers
            for worker in range(self.num_workers):
                start_idx = worker * batch_per_worker
                end_idx = start_idx + batch_per_worker
                worker_actions = actions[start_idx:end_idx]
                for i in range(len(worker_actions)):
             # Her uzay aracı için simülasyon
                    for i in range(len(actions)):
                    # Aksiyonu tensor'a çevir ve doğru cihaza taşı
                        if isinstance(actions[i], np.ndarray):
                            action = torch.from_numpy(actions[i]).float().to(self.device)
                        else:
                            action = actions[i].to(self.device)
                        
                        # Uzay aracını güncelle
                        self.spacecraft[i].apply_action(action)
                        self.spacecraft[i].update(self.dt)  # time_step yerine dt kullan
                        
                        # Yeni durumu al
                        next_state = self.get_state()[i]
                        
                        # Ödülü hesapla
                        reward = float(self._calculate_reward(i))  # float'a çevir
                        
                        # Episode'un bitip bitmediğini kontrol et
                        done = bool(self._check_done(i))  # bool'a çevir
                        
                        # Sonuçları listeye ekle
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done)
                        infos.append({})
                    
                    # Sonuçları tensor'a çevir
                    next_states = torch.stack(next_states)
                    rewards = torch.tensor(rewards, device=self.device)
                    dones = torch.tensor(dones, device=self.device)
                    
                    return next_states, rewards, dones, infos
                        
        except Exception as e:
            print(f"Step batch hatası: {e}")
            traceback.print_exc()
            return (torch.zeros((len(actions), self.state_size), device=self.device),
                torch.zeros(len(actions), device=self.device),
                torch.ones(len(actions), dtype=torch.bool, device=self.device),
                [{} for _ in range(len(actions))])
    
    def get_state_tensor(self):
        """Mevcut durumu tensor olarak al"""
        # Uzay aracı durumu
        spacecraft_state = torch.cat([
            self.spacecraft.position,
            self.spacecraft.velocity,
            torch.tensor([self.spacecraft.fuel_mass], device=self.device)
        ]).unsqueeze(0)  # [1, 5] boyutuna dönüştür

        # Gezegen durumları
        planet_states = []
        for planet in self.planets:
            planet_state = torch.cat([
                planet.position,
                planet.velocity
            ]).unsqueeze(0)  # [1, 4] boyutuna dönüştür
            planet_states.append(planet_state)

        # Tüm gezegen durumlarını birleştir
        all_planet_states = torch.cat(planet_states, dim=0) if planet_states else torch.zeros((0, 4), device=self.device)

        # Tüm durumları birleştir
        state = torch.cat([spacecraft_state, all_planet_states.reshape(-1).unsqueeze(0)], dim=1)
        print(f"Hesaplanan state boyutu: {state}")
        return state.squeeze(0)  # Final boyut: [state_size]

    def update_planet_position(self, planet, dt):
        """Gezegen pozisyonunu Kepler yasalarına göre güncelle"""
        if isinstance(planet, str):
            planet = self.planets[planet]
            
        params = self.orbital_parameters[planet.name]
        
        # Mevcut zaman açısını hesapla
        current_time = self.time % params['orbital_period']
        mean_motion = 2 * np.pi / params['orbital_period']
        
        # Batch boyutunu kontrol et ve mean_anomaly'yi uygun şekilde hesapla
        if isinstance(self.time, torch.Tensor) and self.time.dim() > 0:
            # Batch durumu
            mean_anomaly = mean_motion * current_time + params['initial_angle']
            mean_anomaly = mean_anomaly.to(device=self.device, dtype=torch.float32)
        else:
            # Tekil durum
            mean_anomaly = torch.tensor(mean_motion * current_time + params['initialangle'], 
                                    device=self.device, dtype=torch.float32)
        
        # Eksantrik anomaliyi hesapla (Newton-Raphson metodu)
        e = torch.tensor(params['eccentricity'], device=self.device, dtype=torch.float32)
        E = mean_anomaly.clone()
        for _ in range(5):  # 5 iterasyon genellikle yeterli
            E = E - (E - e * torch.sin(E) - mean_anomaly) / (1 - e * torch.cos(E))
        
        # Gerçek anomaliyi hesapla
        true_anomaly = 2 * torch.atan(torch.sqrt((1 + e)/(1 - e)) * torch.tan(E/2))
        
        # Yarı-major eksen
        a = torch.tensor(params['semi_major_axis'], device=self.device, dtype=torch.float32)
        
        # Radyal mesafe
        r = a * (1 - e * torch.cos(E))
        
        # Pozisyon ve hız hesaplamaları için boyut kontrolü
        if r.dim() > 0:
            # Batch durumu - pozisyonları güncelle
            planet.position = torch.stack([
                r * torch.cos(true_anomaly),
                r * torch.sin(true_anomaly)
            ], dim=-1)
        else:
            # Tekil durum
            planet.position[0] = r * torch.cos(true_anomaly)
            planet.position[1] = r * torch.sin(true_anomaly)
        
        # Yörüngesel hız
        p = a * (1 - e*e)  # yarı-latus rectum
        velocity_magnitude = torch.sqrt(self.G * self.sun_mass * (2/r - 1/a))
        
        # Hız vektörü - boyut kontrolü ile
        if velocity_magnitude.dim() > 0:
            # Batch durumu
            planet.velocity = torch.stack([
                -velocity_magnitude * torch.sin(true_anomaly),
                velocity_magnitude * torch.cos(true_anomaly) * (1 + e * torch.cos(true_anomaly))
            ], dim=-1)
        else:
            # Tekil durum
            planet.velocity[0] = -velocity_magnitude * torch.sin(true_anomaly)
            planet.velocity[1] = velocity_magnitude * torch.cos(true_anomaly) * (1 + e * torch.cos(true_anomaly))
        
        # Hızı normalize et
        planet.velocity *= torch.sqrt(self.G * self.sun_mass / r)
        
        return planet.position, planet.velocity

    def _create_planets(self) -> Dict[str, Planet]:
        """Gezegenleri oluştur ve başlangıç konumlarını ayarla"""
        planets = {}
        for name, data in PLANETARY_DATA.items():
            planet = Planet(name, data['mass'], data['orbit_radius'])
            if name in self.orbital_parameters:
                params = self.orbital_parameters[name]
                
                # Başlangıç açısını tensor olarak bir kez oluştur
                initial_angle = params['initial_angle'].clone().detach()
                
                # Trigonometrik hesaplamaları bir kez yap
                cos_angle = torch.cos(initial_angle)
                sin_angle = torch.sin(initial_angle)
                
                # Yarıçapı hesapla
                r = params['semi_major_axis'] * (1 - params['eccentricity']**2) / \
                    (1 + params['eccentricity'] * cos_angle)
                
                # Pozisyon vektörünü oluştur
                planet.position = torch.stack([
                    r * cos_angle,
                    r * sin_angle
                ]).to(self.device)
                
                # Hız büyüklüğünü hesapla
                velocity_magnitude = torch.sqrt(self.G * self.sun_mass * 
                                             (2/r - 1/params['semi_major_axis']))
                
                # Hız vektörünü oluştur
                planet.velocity = torch.stack([
                    -velocity_magnitude * sin_angle,
                    velocity_magnitude * cos_angle
                ]).to(self.device)
                
            planets[name] = planet
        
        return planets

    def get_state(self):
        """Güncel durumu döndür - optimize edilmiş versiyon"""
        try:
            # Tüm durumları tek seferde hesapla
            spacecraft_states = torch.stack([
                torch.cat([
                    s.position,
                    s.velocity,
                    torch.tensor([s.fuel_mass], device=self.device)
                ]) for s in self.spacecraft
            ])
            
            # Hedef gezegen durumu - tek seferde
            target_state = torch.cat([
                self.destination.position,
                self.destination.velocity
            ]).repeat(self.num_envs, 1)
            
            # Zaman ve diğer metrikler - tek seferde
            other_metrics = torch.stack([
                self.time,
                torch.zeros_like(self.time)  # veya başka bir metrik
            ], dim=1)
            
            # Tüm durumları birleştir
            states = torch.cat([
                spacecraft_states,
                target_state,
                other_metrics
            ], dim=1)
            
            return states
            
        except Exception as e:
            print(f"State hesaplama hatası: {e}")
            return torch.zeros((self.num_envs, self._state_size), device=self.device)
        
    def _check_collision(self, env_idx: int) -> bool:
       """Çarpışma kontrolü - optimize edilmiş versiyon"""
       try:
           # Uzay aracının pozisyonunu al
           spacecraft_pos = self.spacecraft[env_idx].position
           
           # Güneş'le çarpışma kontrolü
           sun_distance = torch.norm(spacecraft_pos)
           if sun_distance < MIN_SAFE_DISTANCE_SUN:
               return True
           
           # Tüm gezegenlerin pozisyonlarını tek seferde al
           planet_positions = torch.stack([p.position for p in self.planets.values()])
           planet_radii = torch.tensor([p.radius for p in self.planets.values()], device=self.device)
           
           # Mesafeleri tek seferde hesapla
           distances = torch.norm(planet_positions - spacecraft_pos.unsqueeze(0), dim=1)
           
           # Çarpışma kontrolü
           return bool(torch.any(distances <= planet_radii * 1.5))  # 1.5 güvenlik faktörü
           
       except Exception as e:
           print(f"Çarpışma kontrolü hatası: {e}")
           return False

    def reset(self):
        try:
            # CPU'da başlangıç değerlerini ayarla
            earth = self.planets['Earth']
            earth_pos = earth.position.clone().cpu()
            earth_vel = earth.velocity.clone().cpu()
            
            for s in self.spacecraft:
                # CPU'da değerleri ayarla
                s.position = earth_pos.clone()
                s.velocity = earth_vel.clone()
                s.fuel_mass = torch.tensor(1000.0, device='cpu')
                s.hull_integrity = torch.tensor(1.0, device='cpu')
                s.engine_temperature = torch.tensor(293.15, device='cpu')
                s.has_fuel_leak = False
                s.backup_engines = 3
                
                # GPU'ya taşı
                s.position = s.position.to(self.device)
                s.velocity = s.velocity.to(self.device)
                s.fuel_mass = s.fuel_mass.to(self.device)
                s.hull_integrity = s.hull_integrity.to(self.device)
                s.engine_temperature = s.engine_temperature.to(self.device)
            
            # Initial distance'ı ayarla
            self.initial_distance = torch.norm(
                self.spacecraft[0].position - self.destination.position
            )
            
            # Previous distance'ı başlat
            self.previous_distance = torch.full(
                (self.num_envs,), float('inf'), device=self.device
            )
            
            # Zamanı sıfırla
            self.time = torch.zeros(self.num_envs, device=self.device)
            self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
            # State'i döndür
            state = self.get_state()
            return state.cpu().numpy()
            
        except Exception as e:
            print(f"Reset hatası: {str(e)}")
            torch.cuda.empty_cache()
            return np.zeros(self.state_size)
        
    def reset_batch(self, batch_size):
       """Batch halinde çevre sıfırlama"""
       try:
           states = []
           for _ in range(batch_size):
               # Her bir ortam için reset çağır
               state = self.reset()
               
               # numpy array'i tensor'a çevir
               if isinstance(state, np.ndarray):
                   state = torch.from_numpy(state).float()
               
               # GPU'ya taşı
               state = state.to(self.device)
               states.append(state)
           
           # Tüm state'leri birleştir
           return torch.stack(states)
           
       except Exception as e:
           print(f"Reset batch hatası: {e}")
           # Hata durumunda sıfır tensor'u dön
           return torch.zeros((batch_size, self.state_size), device=self.device) 
          
    def step(self, action):
        try:
            # NumPy dizisini tensor'a dönüştür
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float().to(self.device)
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
                    
            # Eylemi normalize et
            normalized_action = torch.clamp(action, -1, 1)
            
            # İtki kuvvetini hesapla
            thrust_direction = normalized_action[:2] if normalized_action.dim() == 1 else normalized_action[:, :2]
            thrust_magnitude = torch.abs(normalized_action[2]) if normalized_action.dim() == 1 else torch.abs(normalized_action[:, 2])
            
            # Yön vektörünü normalize et
            thrust_direction = thrust_direction / (torch.norm(thrust_direction, dim=-1, keepdim=True) + 1e-8)
            
            # Tüm uzay araçları için döngü
            for i in range(self.num_envs):
                # İtki kuvvetini hesapla
                thrust_force = thrust_magnitude[i] * MAX_THRUST * thrust_direction[i] if thrust_direction.dim() > 1 else thrust_magnitude * MAX_THRUST * thrust_direction
                
                # Toplam kuvveti hesapla
                total_force = torch.zeros_like(thrust_force)
                
                # Gezegenlerin çekim kuvvetlerini ekle
                for _, planet in self.planets.items():
                    r = planet.position - self.spacecraft[i].position
                    r_mag = torch.norm(r)
                    if r_mag > 0:
                        force = G * planet.mass * self.spacecraft[i].mass * r / (r_mag ** 3)
                        if force.shape != thrust_force.shape:
                            force = force[:2] if force.dim() == 1 else force[:, :2]
                        total_force = total_force + force
                
                # İtki kuvvetini ekle
                total_force = total_force + thrust_force
                
                # Uzay aracını güncelle
                self.spacecraft[i].apply_force(total_force)
                self.spacecraft[i].update(self.dt)
            
            # Gezegenleri güncelle
            for _, planet in self.planets.items():
                planet.update_position_rk45(self.dt)
            
            # Yeni durumu al
            next_state = self.get_state()
            
            # Ödülü hesapla
            reward = self._calculate_reward()
            
            # Bölümün bitip bitmediğini kontrol et
            done = self._check_episode_end()
            
            return next_state, reward, done, {}
                
        except Exception as e:
            print(f"Simülasyon hatası: {e}")
            traceback.print_exc()
            return (torch.zeros((self.num_envs, self._state_size), device=self.device),
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(True, device=self.device),
                    {})

    def _check_emergencies(self, time_step: float) -> Dict[str, bool | float]:
        """Acil durumları kontrol et"""
        result = {'critical_failure': False, 'penalty': 0.0}
        
        # Motor arızası kontrolü
        if random.random() < MOTOR_FAILURE_PROBABILITY * (time_step / 86400):
            if not self.spacecraft.handle_motor_failure():
                result['critical_failure'] = True
                result['penalty'] = -10000
                return result
        
        # Yakıt sızıntısı kontrolü
        if random.random() < FUEL_LEAK_PROBABILITY * (time_step / 86400):
            if not self.spacecraft.handle_fuel_leak():
                result['critical_failure'] = True
                result['penalty'] = -5000
                return result
        
        # Mikroasteroit çarpışması kontrolü
        if random.random() < MICROASTEROID_COLLISION_PROBABILITY * (time_step / 86400):
            if not self.spacecraft.handle_microasteroid_collision():
                result['critical_failure'] = True
                result['penalty'] = -8000
                return result
        
        return result

    def _calculate_reward(self, spacecraft_index=None):
        """Ödül hesaplama"""
        try:
            # Uzay aracını belirle
            if spacecraft_index is None:
                spacecraft = self.spacecraft[0]  # Tek uzay aracı durumu
            else:
                spacecraft = self.spacecraft[spacecraft_index]
                # Mesafe bazlı ödül
            current_distance = torch.norm(spacecraft.position - self.destination.position)
            distance_reward = -current_distance / AU  # AU'ya göre normalize et
            
            # Hız bazlı ödül
            velocity = torch.norm(spacecraft.velocity)
            optimal_velocity = 30000  # Optimal hız (m/s)
            velocity_reward = -abs(velocity - optimal_velocity) / optimal_velocity
            
            # Yakıt kullanımı bazlı ödül
            fuel_efficiency = spacecraft.fuel_mass / self.max_fuel
            fuel_reward = fuel_efficiency * 0.5  # Yakıt tasarrufu için ödül
            
            # Çarpışma cezası
            collision_penalty = 0.0
            if self._check_collision(spacecraft_index if spacecraft_index is not None else 0):
                collision_penalty = -100.0
            
            # Başarı ödülü
            success_reward = 0.0
            if current_distance < SUCCESS_DISTANCE:
                success_reward = 100.0
            
            # Tüm ödülleri birleştir
            reward_dict = {
                'distance_reward': float(distance_reward),
                'velocity_reward': float(velocity_reward),
                'fuel_reward': float(fuel_reward),
                'collision_penalty': float(collision_penalty),
                'success_reward': float(success_reward)
            }
            
            # Toplam ödülü hesapla
            total_reward = sum(reward_dict.values())
            
            # Ödülü sınırla
            total_reward = max(min(total_reward, 100.0), -100.0)
            
            return total_reward
        except Exception as e:
            print(f"Ödül hesaplama hatası: {e}")
            traceback.print_exc()
            return 0.0  # Hata durumunda varsayılan değer
        
    def _check_done(self, spacecraft_index):
        """Episode'un bitip bitmediğini kontrol et"""
        try:
            spacecraft = self.spacecraft[spacecraft_index]
            
            # Hedef gezegene ulaşma kontrolü
            distance_to_target = torch.norm(spacecraft.position - self.destination.position)
            if float(distance_to_target) < SUCCESS_DISTANCE:  # tensor'ı float'a çevir
                return True
                
            # Yakıt bitti mi?
            if float(spacecraft.fuel_mass) <= 0:  # tensor'ı float'a çevir
                return True
                
            # Çarpışma kontrolü
            if self._check_collision(spacecraft_index):
                return True
                
            # Maksimum adım sayısı kontrolü
            if float(self.current_step[spacecraft_index]) >= self.max_steps:  # tensor'ı float'a çevir
                return True
                
            return False
            
        except Exception as e:
            print(f"Done kontrolü hatası: {e}")
            return True  # Hata durumunda episode'u bitir
       
    def adaptive_time_step(self, min_dt=(86400/2), max_dt=86400*15):  # min 1 gün, max 15 gün
        """Adaptif zaman adımı hesaplama"""
        min_period = float('inf')
        for planet in self.planets.values():
            if planet.name in self.orbital_parameters:
                period = self.orbital_parameters[planet.name]['orbital_period']
                min_period = min(min_period, period)
        
        # Daha büyük zaman adımları kullan
        dt = min(max(min_period / 50, min_dt), max_dt)
        
        # Hedef gezegene yakınken daha küçük adımlar kullan
        distance_to_destination = torch.norm(
            torch.stack([s.position for s in self.spacecraft]) - 
            self.destination.position, dim=1
        )
        if torch.any(distance_to_destination < AU):
            dt = min(dt, 86400)  # 1 gün
        
        return dt

    def _initialize_spacecraft(self):
        """Uzay aracını başlangıç konumuna yerleştir"""
        earth = self.planets['Earth']
        for s in self.spacecraft:
            s.position = earth.position.clone()
            s.velocity = earth.velocity.clone()
        self.max_fuel = self._calculate_minimum_fuel()
        for s in self.spacecraft:
            s.fuel_mass = self.max_fuel

    def _calculate_minimum_fuel(self) -> float:
        """Minimum yakıt miktarını hesapla"""
        r1 = torch.norm(self.planets['Earth'].position)
        r2 = torch.norm(self.destination.position)
        mu = self.G * self.sun_mass
        
        # Hohmann transfer için delta-v hesabı
        delta_v1 = torch.sqrt(mu / r1) * (torch.sqrt(2 * r2 / (r1 + r2)) - 1)  # Parantez düzeltildi
        
        delta_v2 = torch.sqrt(mu / r2) * (1 - torch.sqrt(2 * r1 / (r1 + r2)))
        
        delta_v = delta_v1 + delta_v2
        
        m0 = self.spacecraft[0].dry_mass * 40
        mf = m0 / torch.exp(delta_v / (ISP * g0))
        return float(torch.max(torch.tensor(0.0), m0 - mf))

    def _update_history(self):
        """Geçmiş kayıtlarını güncelle"""
        self.fuel_history.append(torch.tensor([s.fuel_mass for s in self.spacecraft], device=self.device))
        self.distance_history.append(torch.norm(
            torch.stack([s.position for s in self.spacecraft]) - 
            self.planets['Earth'].position, dim=1
        ))
        self.speed_history.append(torch.norm(torch.stack([s.velocity for s in self.spacecraft]), dim=1))
        self.time_history.append(self.time.clone())
        self.acceleration_history.append(
            torch.norm(torch.stack([s.acceleration for s in self.spacecraft]), dim=1))
    
    def _check_episode_end(self):
        """Bölümün bitip bitmediğini kontrol et"""
        # Her simülasyon iin kontrol yap
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for i in range(self.num_envs):
            # Güneş'e çarpma kontrolü
            sun_distance = torch.norm(self.spacecraft[i].position)
            if sun_distance < MIN_SAFE_DISTANCE_SUN * 0.5:
                dones[i] = True
            
            # Hedef gezegene varış kontrolü
            distance_to_destination = torch.norm(
                self.spacecraft[i].position - self.destination.position
            )
            if distance_to_destination < SUCCESS_DISTANCE:
                dones[i] = True
            
            # Yakıt bitti mi kontrolü
            if self.spacecraft[i].fuel_mass <= 0:
                dones[i] = True
            
            # Zaman aşımı kontrolü
            if self.time[i] >= MAX_EPISODE_TIME:
                dones[i] = True
            
            # Kritik hasar kontrolü
            if self.spacecraft[i].hull_damage >= self.spacecraft[i].critical_damage_threshold:
                dones[i] = True
        
        return dones


class TrainingTimeEstimator:
    def __init__(self, num_episodes: int):
        self.num_episodes = num_episodes
        self.start_time = None
        self.completed_episodes = 0
        self.episode_times = deque(maxlen=100)  # Son 100 bölümün sürelerini tut
        
    def start(self):
        """Eğitim zamanlayıcısını başlat"""
        self.start_time = time.time()
        self.completed_episodes = 0
        
    def update(self, episode: int) -> dict:
        """Her bölüm sonunda süre tahminini güncelle"""
        current_time = time.time()
        if self.start_time is None:
            self.start(episode)
            return {}
            
        self.completed_episodes = episode + 1
        episode_time = (current_time - self.start_time) / self.completed_episodes
        self.episode_times.append(episode_time)
        
        # Ortalama bölüm süresini hesapla
        avg_episode_time = sum(self.episode_times) / len(self.episode_times)
        
        # Kalan süreyi hesapla
        remaining_episodes = self.num_episodes - self.completed_episodes
        estimated_remaining_time = remaining_episodes * avg_episode_time
        
        # Toplam süreyi hesapla
        total_time = (current_time - self.start_time) + estimated_remaining_time  # starttime -> start_time düzeltildi
        
        # GPU kullanım bilgisini al
        gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        return {
            'elapsed_time': str(timedelta(seconds=int(current_time - self.start_time))),
            'estimated_remaining': str(timedelta(seconds=int(estimated_remaining_time))),
            'total_estimated_time': str(timedelta(seconds=int(total_time))),
            'progress_percentage': (self.completed_episodes / self.num_episodes) * 100,
            'gpu_usage': f"{gpu_usage * 100:.1f}%"
        }

def train_agent(destination_planet: str, num_episodes: int):
   try:
       # Shared data sözlüğünü başlangıçta tanımla
       shared_data = {
           'running': True,
           'current_metrics': {},
           'current_positions': {}
       }
       
       # Ortam ve genetik SAC ajanını oluştur
       env = SpaceTravelEnv(destination_planet, num_envs=48)
       agent = GeneticSACAgent(env.state_size, env.action_size, population_size=48)
       # Metrikleri kaydet bölümüne ekleyin:
       best_rewards.append(max(episode_rewards) if episode_rewards else 0.0)
       # Eğitim metrikleri
       total_rewards = []
       episode_losses = []
       best_rewards = []
       success_rates = []
       min_distances = []
       fuel_consumptions = []
       step_counts = []
       
       # Zaman tahmincisini başlat
       time_estimator = TrainingTimeEstimator(num_episodes)
       time_estimator.start()
       
       for episode in range(num_episodes):
           states = env.reset_batch(agent.population_size)  # Tüm popülasyonu sıfırla
           episode_rewards = []
           episode_steps = []
           episode_distances = []
           episode_fuels = []
           
           for step in range(MAX_STEPS):
               try:
                   # Batch halinde eylemler al
                   actions = agent.act_batch(states)
                   
                   # Tüm popülasyon için adım at
                   next_states, rewards, dones, _ = env.step_batch(actions)
                   
                   # Deneyimleri kaydet ve öğren
                   agent.remember_batch(states, actions, rewards, next_states, dones)
                   loss = agent.replay()
                   
                   # Metrikleri kaydet
                   episode_rewards.extend(rewards.cpu().numpy())
                   episode_steps.append(step + 1)
                   
                   # Mesafe ve yakıt tüketimini kaydet
                   for i in range(len(env.spacecraft)):
                       distance = torch.norm(env.spacecraft[i].position - env.destination.position)
                       episode_distances.append(float(distance) / AU)
                       episode_fuels.append(float(env.spacecraft[i].fuel_mass))
                   
                   states = next_states
                   
                   # Tüm popülasyon üyeleri bittiyse döngüden çık
                   if all(dones):
                       break
                       
               except Exception as e:
                   print(f"Step hatası: {e}")
                   continue
           
           try:
               # Bölüm metriklerini hesapla
               avg_reward = np.mean(episode_rewards)
               avg_steps = np.mean(episode_steps)
               min_distance = min(episode_distances)
               avg_fuel = np.mean(episode_fuels)
               success_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100
               
               # Metrikleri kaydet
               total_rewards.append(avg_reward)
               if loss is not None and isinstance(loss, dict):
                    loss = loss.get('value_loss', 0.0)  # Doğrudan loss'u güncelle
               else:
                    loss = float(loss) if loss is not None else 0.0

               episode_losses.append(loss)
               step_counts.append(avg_steps)
               min_distances.append(min_distance)
               fuel_consumptions.append(avg_fuel)
               success_rates.append(success_rate)
               
               
               if episode % 2 == 0:
                   time_metrics = time_estimator.update(episode)
                   print("\nEpoch {}/{}".format(episode, num_episodes))
                   print("=" * 50)
                   print(f"Geçen Süre: {time_metrics['elapsed_time']}")
                   print(f"Tahmini Kalan Süre: {time_metrics['estimated_remaining']}")
                   print(f"Toplam Tahmini Süre: {time_metrics['total_estimated_time']}")
                   print(f"İlerleme: %{time_metrics['progress_percentage']:.1f}")
                   print(f"GPU Kullanımı: {time_metrics['gpu_usage']}")
                   print(f"Ortalama Ödül: {avg_reward:.2f}")
                   print(f"Ortalama Kayıp: {loss:.4f}" if loss is not None else "Kayıp: N/A")
                   print(f"Ortalama Adım Sayısı: {avg_steps:.1f}")
                   print(f"Başarı Oranı: {success_rate:.1f}%")
                   print(f"Minimum Mesafe: {min_distance:.2f} AU")
                   print(f"Yakıt Tüketimi: {avg_fuel:.1f} kg")
                   print("=" * 50)
                   
                   # GUI için metrikleri güncelle
                   shared_data['current_metrics'].update({
                       'episode': episode,
                       'reward': avg_reward,
                       'loss': loss if loss is not None else 0.0,
                       'success_rate': success_rate,
                       'min_distance': min_distance,
                       'fuel': avg_fuel
                   })
               
               # Evrim uygula
               agent.evolve()
               
           except Exception as e:
               print(f"Metrik hesaplama hatası: {e}")
               continue
           
       return agent, total_rewards, episode_losses, success_rates, min_distances, fuel_consumptions
           
   except Exception as e:
       print(f"Eğitim hatası: {e}")
       traceback.print_exc()
       return None, [], [], [], [], []

def simulate_journey(env: SpaceTravelEnv, agent: SACAgent):
    state = env.reset()
    spacecraft_positions = [env.spacecraft.position.cpu().numpy()]
    
    # Simlasyon metrikleri
    metrics = {
        'velocities': [float(torch.norm(env.spacecraft.velocity))],
        'accelerations': [0],
        'fuel_levels': [float(env.spacecraft.fuel_mass)],
        'distances_to_target': [float(torch.norm(
            env.spacecraft.position - env.destination.position
        ))],
        'time': [0]
    }
    
    while not env.done:
        # GPU üzerinde action hesaplama
        with torch.no_grad():
            action = agent.act(state)
        
        # Simülasyon admı
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:  # Eğer bölüm bittiyse döngüden çık
           break
        # Pozisyon ve metrik kayıtları
        spacecraft_positions.append(env.spacecraft.position.cpu().numpy())
        metrics['velocities'].append(float(torch.norm(env.spacecraft.velocity)))
        metrics['accelerations'].append(float(torch.norm(env.spacecraft.acceleration)))
        metrics['fuel_levels'].append(float(env.spacecraft.fuel_mass))
        metrics['distances_to_target'].append(float(torch.norm(
            env.spacecraft.position - env.destination.position
        )))
        metrics['time'].append(env.time)
    
    return env, np.array(spacecraft_positions), metrics

def create_animation(env: SpaceTravelEnv, spacecraft_positions: np.ndarray, 
                    metrics: dict, destination_planet_name: str):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f'Uzay Yolculuğu Simülasyonu - Hedef: {destination_planet_name}', 
                color='white', size=12)
    
    # Scaling factor for AU conversion
    scale_factor = 1 / AU
    
    # Calculate maximum distance for plot limits
    max_distance = max(
        np.max(np.abs(spacecraft_positions[:, 0])),
        np.max(np.abs(spacecraft_positions[:, 1])) * scale_factor  # Add closing parenthesis here
    )
    
    # Main plot settings
    ax1.set_xlim(-max_distance * 1.1, max_distance * 1.1)
    ax1.set_ylim(-max_distance * 1.1, max_distance * 1.1)
    ax1.set_xlabel('X Position (AU)')
    ax1.set_ylabel('Y Position (AU)')
    ax1.grid(True, alpha=0.3)
    
    # Create sun
    sun = plt.Circle((0, 0), 0.1, color='yellow')
    ax1.add_artist(sun)
    
    # Planet orbits and Lagrange points
    for name, planet in env.planets.items():
        # Orbit
        orbit = plt.Circle((0, 0), planet.orbit_radius * scale_factor, 
                          fill=False, linestyle='--', alpha=0.3)
        ax1.add_artist(orbit)
        
        # Convert Lagrange points to numpy
        l1_points = torch.stack(planet.l1_positions).cpu().numpy() * scale_factor
        l2_points = torch.stack(planet.l2_positions).cpu().numpy() * scale_factor
        
        # Plot Lagrange points
        ax1.plot(l1_points[:, 0], l1_points[:, 1], 'c.', alpha=0.3, label=f'{name} L1')
        ax1.plot(l2_points[:, 0], l2_points[:, 1], 'm.', alpha=0.3, label=f'{name} L2')
        
        # Influence radius circles
        influence_radius = planet.lagrange.influence_radius * scale_factor
        l1_circle = plt.Circle((l1_points[-1, 0], l1_points[-1, 1]), 
                             influence_radius, color='c', fill=False, alpha=0.2)
        l2_circle = plt.Circle((l2_points[-1, 0], l2_points[-1, 1]), 
                             influence_radius, color='m', fill=False, alpha=0.2)
        ax1.add_artist(l1_circle)
        ax1.add_artist(l2_circle)
    
    # Spacecraft trail
    trail, = ax1.plot([], [], 'r-', alpha=0.5)
    spacecraft, = ax1.plot([], [], 'ro')
    
    # Metrics plot
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Speed (km/s)')
    ax2.grid(True, alpha=0.3)
    speed_line, = ax2.plot([], [])
    
    ax1.legend(loc='upper right', fontsize='small')
    
    def animate(frame):
        # Update spacecraft position
        pos = spacecraft_positions[frame] * scale_factor
        spacecraft.set_data(pos[0], pos[1])
        
        # Update trail
        trail.set_data(spacecraft_positions[:frame+1, 0] * scale_factor,
                      spacecraft_positions[:frame+1, 1] * scale_factor)
        
        # Update speed plot
        times = np.array(metrics['time'][:frame+1]) / (24*3600)  # Convert to days
        speeds = np.array(metrics['velocities'][:frame+1]) / 1000  # Convert to km/s
        speed_line.set_data(times, speeds)
        ax2.relim()
        ax2.autoscale_view()
        
        return spacecraft, trail, speed_line
    
    ani = animation.FuncAnimation(fig, animate, frames=len(spacecraft_positions),
                                interval=50, blit=True)
    plt.close()
    return ani

def plot_training_metrics(scores: List[float], losses: List[float], epsilons: List[float]):
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    ax1.plot(scores)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epsilons)
    ax3.set_title('Exploration Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

class SpaceSimulationGUI:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Uzay Aracı Simülasyonu")
        
        # Renkler
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.YELLOW = (255, 255, 0)  # Güneş rengi
        self.RED = (255, 0, 0)      # Uzay aracı rengi
        self.BLUE = (0, 0, 255)     # Hedef gezegen rengi
        self.GREEN = (0, 255, 0)    # Diğer gezegen rengi
        self.GRAY = (128, 128, 128)  # Yörünge rengi
        
        # Ekran bölümleri
        self.sim_width = int(width * 0.8)  # Simülasyon alanı - %80
        self.panel_width = width - self.sim_width  # Yan panel - %20
        
        # Font ve UI elemanları
        self.title_font = pygame.font.Font(None, 48)  # Başlık fontu
        self.font = pygame.font.Font(None, 36)       # Normal font
        self.small_font = pygame.font.Font(None, 24)  # Küçük font
        
        # Simülasyon kontrolleri
        self.paused = False  # Duraklatma durumu
        self.current_sim_index = 0  # Aktif simülasyon indeksi
        self.show_all_sims = False  # Tüm simülasyonları göster
        self.zoom_level = 1.0  # Yakınlaştırma seviyesi
        self.pan_offset = [0, 0]  # Kaydırma konumu
        
        # Performans optimizasyonu
        self.clock = pygame.time.Clock()
        self.fps = 60  # Saniyedeki kare sayısı
        self.surface_cache = {}  # Yüzey önbelleği
        self.trail_points = {i: [] for i in range(24)}  # Her simülasyon için iz noktaları
        self.max_trail_length = 1000  # Maksimum iz uzunluğu
        
        # Düğmeler
        self.buttons = {
            'next_sim': pygame.Rect(width - 150, height - 100, 100, 40),  # Sonraki simülasyon
            'prev_sim': pygame.Rect(width - 270, height - 100, 100, 40),  # Önceki simülasyon
            'toggle_all': pygame.Rect(width - 390, height - 100, 100, 40),  # Tümünü göster/gizle
            'reset_view': pygame.Rect(width - 150, height - 160, 100, 40)  # Görünümü sıfırla
        }
        
        # Metrik geçmişi
        self.metric_history = {
            'hz': deque(maxlen=100),  # Son 100 hız değeri
            'yakıt': deque(maxlen=100),  # Son 100 yakıt değeri
            'mesafe': deque(maxlen=100),  # Son 100 mesafe değeri
            'ödül': deque(maxlen=100)  # Son 100 ödül değeri
        }
        
        # Ölçekleme faktörü - 60 AU'luk görüş alanı
        self.scale = width / (60 * AU)
        
        # Başlangıç pozisyonları
        self.center_x = width // 2  # Merkez X koordinatı
        self.center_y = height // 2  # Merkez Y koordinatı

    def world_to_screen(self, pos):
        """Dünya koordinatlarını ekran koordinatlarına dönüştür"""
        x = (pos[0] * self.scale * self.zoom_level + self.pan_offset[0]) + self.sim_width/2
        y = (pos[1] * self.scale * self.zoom_level + self.pan_offset[1]) + self.height/2
        return int(x), int(y)

    def draw_side_panel(self, metrics):
        """Yan paneli çiz - Metrik göstergeleri"""
        # Panel arkaplanı
        panel_rect = pygame.Rect(self.sim_width, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, self.BLACK, panel_rect)
        pygame.draw.line(self.screen, self.WHITE, (self.sim_width, 0), (self.sim_width, self.height))
        
        # Panel başlığı
        title = self.title_font.render("Metrikler", True, self.WHITE)
        self.screen.blit(title, (self.sim_width + 20, 20))
        
        # Metrikleri göster
        y = 100
        for key, value in metrics.items():
            # Metrik geçmişini güncelle
            self.metric_history[key].append(float(value))
            
            # Metrik değerini göster
            text = self.font.render(f"{key}: {value}", True, self.WHITE)
            self.screen.blit(text, (self.sim_width + 20, y))
            
            # Mini grafik çiz
            self.draw_mini_graph(key, self.sim_width + 20, y + 30)
            y += 100
        
        # Kontrol düğmeleri
        self.draw_buttons()
        
        # Simülasyon bilgisi
        sim_info = self.font.render(f"Simülasyon {self.current_sim_index + 1}/24", True, self.WHITE)
        self.screen.blit(sim_info, (self.sim_width + 20, self.height - 200))

    def draw_mini_graph(self, metric_key, x, y):
        """Mini grafik çiz - Metrik geçmişi görselleştirmesi"""
        if len(self.metric_history[metric_key]) < 2:
            return
            
        values = list(self.metric_history[metric_key])
        max_val = max(values)
        min_val = min(values)
        range_val = max_val - min_val or 1
        
        points = []
        for i, val in enumerate(values):
            px = x + (i * 200 / len(values))
            py = y + 50 - ((val - min_val) * 50 / range_val)
            points.append((int(px), int(py)))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.GREEN, False, points, 2)

    def draw_buttons(self):
        """Kontrol düğmelerini çiz"""
        for name, rect in self.buttons.items():
            color = self.GRAY if name == 'toggle_all' and self.show_all_sims else self.BLUE
            pygame.draw.rect(self.screen, color, rect)
            
            text = None
            if name == 'next_sim':
                text = self.font.render("→", True, self.WHITE)  # Sonraki
            elif name == 'prev_sim':
                text = self.font.render("←", True, self.WHITE)  # Önceki
            elif name == 'toggle_all':
                text = self.small_font.render("Tümü", True, self.WHITE)  # Tümünü göster/gizle
            elif name == 'reset_view':
                text = self.small_font.render("Sıfırla", True, self.WHITE)  # Görünümü sıfırla
                
            if text:
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

    def draw_simulation(self, env, metrics):
        """Simülasyonu çiz - Ana görselleştirme fonksiyonu"""
        # Ekranı temizle
        self.screen.fill(self.BLACK)
        
        # Güneş'i çiz
        sun_pos = (self.center_x + self.pan_offset[0], 
                  self.center_y + self.pan_offset[1])
        pygame.draw.circle(self.screen, self.YELLOW, sun_pos, 
                         int(20 * self.zoom_level))
        
        # Gezegenleri ve yörüngelerini çiz
        for name, planet in env.planets.items():
            # Yörünge
            orbit_radius = int(planet.orbit_radius * self.scale * self.zoom_level)
            pygame.draw.circle(self.screen, self.GRAY, sun_pos, orbit_radius, 1)
            
            # Gezegen pozisyonu
            pos = planet.position.cpu().numpy()
            screen_x = self.center_x + int(pos[0] * self.scale * self.zoom_level) + self.pan_offset[0]
            screen_y = self.center_y + int(pos[1] * self.scale * self.zoom_level) + self.pan_offset[1]
            
            # Gezegeni çiz
            color = self.BLUE if name == env.destination.name else self.GREEN
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 
                             int(10 * self.zoom_level))
            
            # Gezegen adını yaz
            text = self.small_font.render(name, True, self.WHITE)
            self.screen.blit(text, (screen_x + 10, screen_y - 10))
        
        # Uzay araçlarını çiz
        if self.show_all_sims:
            for i, spacecraft in enumerate(env.spacecraft):
                self.draw_spacecraft(spacecraft, i, alpha=128)  # Yarı saydam
        else:
            self.draw_spacecraft(env.spacecraft[self.current_sim_index], 
                               self.current_sim_index)
        
        # Yan paneli çiz
        self.draw_side_panel(metrics)
        
        # Ekranı güncelle
        pygame.display.flip()

    def draw_spacecraft(self, spacecraft, index, alpha=255):
        """Uzay aracını çiz - İz ve durum göstergeleri"""
        pos = spacecraft.position.cpu().numpy()
        screen_x = self.center_x + int(pos[0] * self.scale * self.zoom_level) + self.pan_offset[0]
        screen_y = self.center_y + int(pos[1] * self.scale * self.zoom_level) + self.pan_offset[1]
        
        # İz noktalarını güncelle
        self.trail_points[index].append((screen_x, screen_y))
        if len(self.trail_points[index]) > self.max_trail_length:
            self.trail_points[index].pop(0)
        
        # İzi çiz
        if len(self.trail_points[index]) > 1:
            pygame.draw.lines(self.screen, (*self.RED[:3], alpha), False, 
                            self.trail_points[index], int(2 * self.zoom_level))
        
        # Uzay aracını çiz
        pygame.draw.circle(self.screen, (*self.RED[:3], alpha), 
                         (screen_x, screen_y), int(5 * self.zoom_level))

    def handle_events(self):
        """Kullanıcı girdilerini işle - Kontrol ve etkileşim"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused  # Duraklat/Devam et
                elif event.key == pygame.K_r:
                    self.zoom_level = 1.0  # Yaknlaştırmayı sıfırla
                    self.pan_offset = [0, 0]  # Kaydırmayı sıfırla
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Fare t��klamalarını işle
                pass

    def run_simulation(self, env, agent):
        """Ana simülasyon döngüs"""
        running = True
        
        while running:
            running = self.handle_events()
            
            if not self.paused:
                # Simülasyon adım
                state = env.get_state()
                with torch.no_grad():
                    action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                state = next_state 
                # Metrikleri güncelle
                metrics = {
                    'hız': f"{float(torch.norm(env.spacecraft[self.current_sim_index].velocity))/1000:.1f} km/s",
                    'yakıt': f"{float(env.spacecraft[self.current_sim_index].fuel_mass):.0f} kg",
                    'mesafe': f"{float(torch.norm(env.spacecraft[self.current_sim_index].position - env.destination.position))/AU:.2f} AU",
                    'ödül': f"{float(reward[self.current_sim_index]):.1f}",
                    'info': info  # info'yu ekle
                }
                
                # Ekranı güncelle
                self.screen.fill(self.BLACK)
                self.draw_simulation(env, metrics)
                self.draw_side_panel(metrics)
                pygame.display.flip()
                
                if done.any():
                    env.reset()
            
            self.clock.tick(self.fps)
        
        pygame.quit()

    def update(self, metrics):
        self.screen.fill(self.BLACK)
        # Metrik bilgilerini ekrana yaz
        font = pygame.font.Font(None, 36)
        y = 10
        for key, value in metrics.items():
            text = font.render(f"{key}: {value:.2f}", True, self.WHITE)
            self.screen.blit(text, (10, y))
            y += 40
        pygame.display.flip()

def run_gui_process(shared_data: Dict[str, Any]):
    """GUI process'ini çalıştıran fonksiyon"""
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    while shared_data['running']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared_data['running'] = False

        # Ekranı temizle
        screen.fill((0, 0, 0))

        # Mevcut metrikleri göster
        if 'current_metrics' in shared_data:
            y = 10
            for key, value in shared_data['current_metrics'].items():
                text = font.render(f"{key}: {value}", True, (255, 255, 255))
                screen.blit(text, (10, y))
                y += 30

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

def train_with_visualization():
    """Eğitim sürecini görselleştirmeyle birlikte yürüten ana fonksiyon"""
    # Shared data dictionary'sini başlangıçta tanımla
    shared_data = {
        'running': True,
        'paused': False,
        'current_metrics': {},
        'current_positions': {}
    }
    
    # GUI process'ini başlat
    gui_process = multiprocessing.Process(target=run_gui_process, args=(shared_data,))
    gui_process.start()
    
    try:
        env = SpaceTravelEnv("Neptune")
        agent = SACAgent(env.state_size, env.action_size)
        
        # Eğitim döngüsü
        for episode in range(NUM_EPISODES):
            state = env.reset()
            total_reward = 0
            
            for step in range(MAX_STEPS):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                total_reward += reward
                
                # GUI için metrikleri güncelle
                shared_data['current_metrics'] = {
                    'episode': episode,
                    'step': step,
                    'reward': total_reward,
                    'epsilon': agent.epsilon
                }
                shared_data['current_positions'] = {
                    'spacecraft': env.spacecraft[0].position.tolist(),
                    'destination': env.destination.position.tolist()
                }
                
                if done:
                    break
            
            # Her 10 bölümde bir hedef modeli güncelle
            if episode % 10 == 0:
                agent.save_model(f"sac_model_episode_{episode}.pth")
            
            print(f"Bölüm: {episode}, Toplam Ödül: {total_reward}, Epsilon: {agent.epsilon}")
        
    except Exception as e:
        print(f"Eğitim hatası: {e}")
    finally:
        shared_data['running'] = False
        if gui_process.is_alive():
            gui_process.join()

def draw_simulation(env, screen):
    """Simülasyonu ekrana çiz"""
    # Renk tanımlamaları
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0) 
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    
    # Ekran boyutları
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    
    # Ölçekleme faktörü
    SCALE = 100
    
    screen.fill(BLACK)
    
    # Güneş'i çiz
    pygame.draw.circle(screen, YELLOW, 
                     (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), 20)
    
    # Gezegenleri çiz
    for name, planet in env.planets.items():
        pos = planet.position.cpu().numpy()
        screen_x = SCREEN_WIDTH//2 + int(pos[0] / AU * SCALE)
        screen_y = SCREEN_HEIGHT//2 + int(pos[1] / AU * SCALE)
        
        color = BLUE if name != env.destination.name else RED
        pygame.draw.circle(screen, color, (screen_x, screen_y), 5)
    
    # Uzay aracını çiz
    for spacecraft in env.spacecraft:
        pos = spacecraft.position.cpu().numpy()
        screen_x = SCREEN_WIDTH//2 + int(pos[0] / AU * SCALE)
        screen_y = SCREEN_HEIGHT//2 + int(pos[1] / AU * SCALE)
        pygame.draw.circle(screen, WHITE, (screen_x, screen_y), 3)
    
    pygame.display.flip()

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.operation_count = 0
        self.last_time = self.start_time
        self.sps_history = []  # Steps per second
        self.flops_history = []  # Floating point operations per second
        
    def update(self, num_operations):
        current_time = time.time()
        self.frame_count += 1
        self.operation_count += num_operations
        
        if current_time - self.last_time >= 1.0:
            # Hesaplama hızı metrikleri
            sps = self.frame_count / (current_time - self.last_time)  # Saniyedeki adım sayısı
            flops = self.operation_count / (current_time - self.last_time)  # Saniyedeki işlem sayısı
            
            self.sps_history.append(sps)
            self.flops_history.append(flops)
            
            # Ekrana yazdır
            print(f"\rAdım/s: {sps:.1f} | "
                  f"İşlem/s: {flops/1e9:.2f} GFlops | "
                  f"Toplam İşlem: {self.operation_count/1e9:.1f}G", end="")
            
            # Sayaçları sıfırla
            self.frame_count = 0
            self.operation_count = 0
            self.last_time = current_time

def main():
    try:
        DESTINATION_PLANET = 'Neptune'
        POPULATION_SIZE = 128  # Arttırıldı
        NUM_EPISODES = 50000 
        BATCH_SIZE = 2048  
        NUM_WORKERS = 16
        # GPU optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False # Hız için deterministik olmayan mod
        torch.cuda.empty_cache()
        
        print("\nSimülasyon başlatılıyor...")
        env = SpaceTravelEnv(DESTINATION_PLANET, num_envs=POPULATION_SIZE,num_workers=NUM_WORKERS)
        print("Ortam başarıyla oluşturuldu")
        
        agent = LongTermGeneticSACAgent(env.state_size, env.action_size,
                                      population_size=POPULATION_SIZE)
        print("Ajan başarıyla oluşturuldu")
        
        # Çoklu GPU desteği
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPU kullanılıyor")
            agent = torch.nn.DataParallel(agent)
        
        # Gelişmiş performans monitörü başlat
        monitor = PerformanceMonitor()
        monitor.gpu_metrics = {i: [] for i in range(torch.cuda.device_count())}
        
        # CUDA Stream oluştur
        stream = torch.cuda.Stream()
        
        # Zaman tahmincisini başlat
        time_estimator = TrainingTimeEstimator(NUM_EPISODES)
        time_estimator.start()
        
        # Gradient scaler oluştur
        scaler = torch.amp.GradScaler()
        
        print("\nEğitim başlıyor...")
        for episode in tqdm(range(NUM_EPISODES), desc="Eğitim İlerlemesi"):
            with torch.cuda.stream(stream):
                state = env.reset()
                episode_reward = torch.zeros(POPULATION_SIZE, device=agent.device)
                done = torch.zeros(POPULATION_SIZE, dtype=torch.bool, device=agent.device)
                step = 0
                max_steps = 500
                success_found = False
                
                while not success_found and step < max_steps:
                    # İşlem sayısını tahmin et ve GPU kullanımını izle
                    num_operations = 0
                    for gpu_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(gpu_id):
                            for layer in agent.module.population[0].sac.actor.modules() if torch.cuda.device_count() > 1 else agent.population[0].sac.actor.modules():
                                if isinstance(layer, torch.nn.Linear):
                                    input_size = layer.weight.shape[1]
                                    output_size = layer.weight.shape[0]
                                    num_operations += 2 * input_size * output_size * POPULATION_SIZE
                            
                            # GPU metriklerini kaydet
                            monitor.gpu_metrics[gpu_id].append(torch.cuda.memory_allocated(gpu_id))

                    # Toplam işlem sayısına batch işlemlerini ekle
                    num_operations += (
                        POPULATION_SIZE *
                        agent.module.action_size if torch.cuda.device_count() > 1 else agent.action_size * 
                        agent.module.state_size if torch.cuda.device_count() > 1 else agent.state_size * 
                        2 *
                        torch.cuda.device_count()  # GPU sayısı ile çarp
                    )
                    
                    # Mixed precision training ve paralel işlem
                    with torch.amp.autocast('cuda'):
                        action = agent.act_batch(state)
                        next_state, reward, done, info = env.step_batch(action)
                        
                        # Fitness skorlarını güncelle
                        for i in range(POPULATION_SIZE):
                            if torch.cuda.device_count() > 1:
                                agent.module.population[i].update_fitness(reward[i].item())
                            else:
                                agent.population[i].update_fitness(reward[i].item())
                                
                            if info[i].get('success', False):
                                success_found = True
                                print(f"\nBaşarı! Uzay aracı {i} hedefe ulaştı!")
                                if torch.cuda.device_count() > 1:
                                    agent.module.save_successful_agent(i, f'successful_agent_{episode}_{i}.pth')
                                else:
                                    agent.save_successful_agent(i, f'successful_agent_{episode}_{i}.pth')
                        
                        # Deneyimi kaydet ve öğren
                        if torch.cuda.device_count() > 1:
                            agent.module.remember_batch(state, action, reward, next_state, done)
                            if len(agent.module.memory) >= BATCH_SIZE:
                                loss = agent.module.replay()
                                if isinstance(loss, torch.Tensor):
                                    scaler.scale(loss).backward()
                                    scaler.step(agent.module.optimizer)
                                    scaler.update()
                        else:
                            agent.remember_batch(state, action, reward, next_state, done)
                            if len(agent.memory) >= BATCH_SIZE:
                                loss = agent.replay()
                                if isinstance(loss, torch.Tensor):
                                    scaler.scale(loss).backward()
                                    scaler.step(agent.optimizer)
                                    scaler.update()
                        
                        state = next_state
                        episode_reward += reward
                    
                    # Performans monitörünü güncelle
                    monitor.update(num_operations)
                    
                    step += 1
                    
                    # Her 100 adımda bir durum raporu
                    if step % 100 == 0:
                        successful_count = sum(1 for i in range(POPULATION_SIZE) 
                                            if info[i].get('success', False))
                        print(f"\nAdım: {step}, Başarılı: {successful_count}/{POPULATION_SIZE}")
            
            # Episode sonu işlemleri
            if torch.cuda.device_count() > 1:
                agent.module.evolve()
            else:
                agent.evolve()
            mean_reward = episode_reward.mean().item()
            # Her 10 episode'da bir detaylı rapor
            if (episode + 1) % 10 == 0:
                time_metrics = time_estimator.update(episode)
                avg_fps = np.mean(monitor.fps_history[-10:])
                avg_flops = np.mean(monitor.flops_history[-10:])
                
                print("\n" + "="*50)
                print(f"Episode: {episode + 1}/{NUM_EPISODES}")
                print(f"Ortalama Ödül: {mean_reward:.2f}")
                print(f"Geçen Süre: {time_metrics['elapsed_time']}")
                print(f"Tahmini Kalan Süre: {time_metrics['estimated_remaining']}")
                print(f"İlerleme: %{time_metrics['progress_percentage']:.1f}")
                print(f"Ortalama SPS: {avg_fps:.2f}")
                print(f"Ortalama GFLOPS: {avg_flops/1e9:.2f}")
                print(f"GPU Kullanımı: {time_metrics['gpu_usage']}")
                print("="*50)
                
                # Checkpoint kaydet
                agent.save_population(f'checkpoint_{episode + 1}.pth')
                
                # Belleği temizle
                if episode % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Eğitim sonu istatistikleri
        print("\nEğitim tamamlandı!")
        print(f"Toplam süre: {time_estimator.update(NUM_EPISODES-1)['elapsed_time']}")
        print(f"Ortalama SPS: {np.mean(monitor.fps_history):.2f}")
        print(f"Ortalama GFLOPS: {np.mean(monitor.flops_history)/1e9:.2f}")
        
        # Son modeli kaydet
        agent.save_population('final_model.pth')
        
        # Eğitimi başlat
        trained_agent, rewards, losses, success_rates, distances, fuels = train_agent(DESTINATION_PLANET, NUM_EPISODES)
        trained_agent.save_population('trained_model.pth')
        # Simülasyon GUI'sini başlat
        env = SpaceTravelEnv(DESTINATION_PLANET)
        gui = SpaceSimulationGUI()
        gui.run_simulation(env, trained_agent)
        
        # Eğitim metriklerini görselleştir
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.plot(rewards)
        plt.title('Ortalama Ödüller')
        plt.xlabel('Episode')
        
        plt.subplot(232)
        plt.plot(losses)
        plt.title('Eğitim Kaybı')
        plt.xlabel('Episode')
        
        plt.subplot(233)
        plt.plot(success_rates)
        plt.title('Başarı Oranı (%)')
        plt.xlabel('Episode')
        
        plt.subplot(234)
        plt.plot(distances)
        plt.title('Minimum Mesafe (AU)')
        plt.xlabel('Episode')
        
        plt.subplot(235)
        plt.plot(fuels)
        plt.title('Yakıt Tüketimi (kg)')
        plt.xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        print("\nMetrikler 'training_metrics.png' dosyasına kaydedildi.")
        
    except Exception as e:
        print(f"\nHata oluştu: {e}")
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        plt.close('all')
        print("\nProgram sonlandırıldı.")

if __name__ == "__main__":
    main()