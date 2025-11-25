#!/usr/bin/env python3
"""
Modèle GRU (Gated Recurrent Unit) pour la prédiction de la fréquence cardiaque.

Ce modèle est une alternative plus efficace au LSTM. Il utilise le même format d'entrée
mais remplace les cellules LSTM par des cellules GRU, qui ont moins de paramètres
et convergent souvent plus rapidement.

Architecture:
    Input: Concatenate[speed, altitude] + gender (répété temporellement)
    GRU: 2 couches (par défaut)
    Output: Séquence de fréquence cardiaque [batch, 500, 1]
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class HeartRateGRU(nn.Module):
    """
    Modèle GRU pour la prédiction de séries temporelles (Fréquence Cardiaque).
    
    Architecture:
        1. Concatenate speed + altitude + gender → [batch, seq_len, 3]
        2. GRU layers avec dropout
        3. Fully connected layer → [batch, seq_len, 1]
    """
    
    def __init__(
        self, 
        input_size=3,           # speed + altitude + gender
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2,
        bidirectional=False
    ):
        """
        Initialisation du modèle GRU.
        
        Args:
            input_size: Nombre de features d'entrée (défaut: 3)
            hidden_size: Dimension cachée du GRU (défaut: 64)
            num_layers: Nombre de couches GRU (défaut: 2)
            dropout: Probabilité de dropout (défaut: 0.2)
            bidirectional: Utiliser un GRU bidirectionnel (défaut: False)
        """
        super(HeartRateGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Couches GRU
        # Note: nn.GRU n'a pas de cell state, contrairement au LSTM
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Couche de Dropout explicite après le GRU
        self.dropout_layer = nn.Dropout(dropout)
        
        # Couche de sortie (Fully Connected)
        # Si bidirectionnel, la sortie est 2x la taille cachée
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(gru_output_size, 1)
        
    def forward(self, speed, altitude, gender, original_lengths=None):
        """
        Passe avant (Forward pass).
        
        Args:
            speed: [batch, seq_len, 1] - Vitesse normalisée
            altitude: [batch, seq_len, 1] - Altitude normalisée
            gender: [batch, 1] - Genre binaire
            original_lengths: [batch, 1] - Longueurs originales (optionnel pour packing)
        
        Returns:
            heart_rate_pred: [batch, seq_len, 1] - BPM prédit
        """
        batch_size, seq_len, _ = speed.shape
        
        # Expansion du genre pour correspondre à la séquence: [batch, 1] → [batch, seq_len, 1]
        gender_expanded = gender.unsqueeze(1).expand(batch_size, seq_len, 1)
        
        # Concaténation des features: [batch, seq_len, 3]
        x = torch.cat([speed, altitude, gender_expanded], dim=2)
        
        # GRU Forward pass
        # Le GRU retourne: output, h_n (pas de c_n comme dans le LSTM)
        # output: [batch, seq_len, hidden_size * num_directions]
        gru_out, h_n = self.gru(x)
        
        # Application du dropout sur les sorties de la séquence
        gru_out = self.dropout_layer(gru_out)
        
        # Projection vers la dimension de sortie (1 BPM): [batch, seq_len, 1]
        heart_rate_pred = self.fc(gru_out)
        
        return heart_rate_pred
    
    def count_parameters(self):
        """Compte les paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WorkoutDataset(Dataset):
    """
    Dataset PyTorch pour les données d'entraînement.
    Identique à la version LSTM pour assurer la compatibilité.
    """
    
    def __init__(self, data_dict):
        self.speed = data_dict['speed']
        self.altitude = data_dict['altitude']
        self.heart_rate = data_dict['heart_rate']
        self.gender = data_dict['gender']
        self.userId = data_dict['userId']
        self.original_lengths = data_dict['original_lengths']
        self.n_samples = len(self.speed)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return (
            self.speed[idx],
            self.altitude[idx],
            self.gender[idx],
            self.heart_rate[idx],
            self.original_lengths[idx]
        )


# Example usage and model check
if __name__ == '__main__':
    print("="*80)
    print("ROBUST GRU MODEL FOR HEART RATE PREDICTION")
    print("="*80)
    
    # Création du modèle
    # Configuration un peu plus robuste par défaut (hidden_size plus large ou bidirectionnel)
    model = HeartRateGRU(
        input_size=3,
        hidden_size=64, 
        num_layers=2,
        dropout=0.2,
        bidirectional=True  # Testons en mode bidirectionnel pour la robustesse
    )
    
    print(f"\nModel Architecture (GRU):")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test avec dummy data
    batch_size = 4
    seq_len = 500
    
    dummy_speed = torch.randn(batch_size, seq_len, 1)
    dummy_altitude = torch.randn(batch_size, seq_len, 1)
    dummy_gender = torch.randint(0, 2, (batch_size, 1)).float()
    dummy_lengths = torch.randint(50, seq_len, (batch_size, 1))
    
    print(f"\n" + "="*80)
    print("TEST FORWARD PASS")
    print("="*80)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_speed, dummy_altitude, dummy_gender, dummy_lengths)
    
    print(f"Input shapes:")
    print(f"  Input features: [batch={batch_size}, seq={seq_len}, feat=3]")
    print(f"\nOutput shape: {output.shape}")
    
    # Vérification simple
    assert output.shape == (batch_size, seq_len, 1), "Erreur de dimension en sortie"
    print("\n✓ Model initialized and tested successfully!")
    print("  Le GRU gère correctement les états cachés sans cell states.")