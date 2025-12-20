import torch
import matplotlib.pyplot as plt
import numpy as np
from problems.op.problem_op import OP
from utils import load_model
import argparse

def plot_op_solution(model_path, n_samples=4, graph_size=20):
    """Visualizza soluzioni del modello per l'Orienteering Problem"""
    
    # Carica il modello
    print(f"Caricamento modello da {model_path}...")
    model, _ = load_model(model_path)
    model.eval()
    model.set_decode_type('greedy')  # IMPORTANTE!
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Genera istanze OP
    print("Generazione dati...")
    problem = OP
    dataset = problem.make_dataset(
        size=graph_size, 
        num_samples=n_samples,
        distribution='unif'
    )
    
    # Converti in batch tensors
    batch = {
        'depot': torch.stack([d['depot'] for d in dataset]).to(device),
        'loc': torch.stack([d['loc'] for d in dataset]).to(device),
        'prize': torch.stack([d['prize'] for d in dataset]).to(device),
        'max_length': torch.stack([d['max_length'] for d in dataset]).to(device)
    }
    
    # Risolvi con il modello
    print("Generazione soluzioni...")
    with torch.no_grad():
        cost, log_p, pi = model(batch, return_pi=True)
    
    # Sposta risultati su CPU
    pi = pi.cpu()
    cost = cost.cpu()
    
    # Crea una figura con subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(min(n_samples, 4)):
        ax = axes[idx]
        
        # Estrai dati dell'istanza
        depot = dataset[idx]['depot'].numpy()
        loc = dataset[idx]['loc'].numpy()
        prize = dataset[idx]['prize'].numpy()
        max_length = dataset[idx]['max_length'].item()
        
        # Estrai il tour
        tour = pi[idx].numpy()
        
        # Plotta tutti i nodi in grigio
        ax.scatter(loc[:, 0], loc[:, 1], c='lightgray', s=100, 
                  alpha=0.5, label='Non visitati', zorder=1)
        
        # Plotta il depot
        ax.scatter(depot[0], depot[1], c='red', s=300, marker='s', 
                  label='Depot', zorder=5, edgecolors='black', linewidth=2)
        
        # Costruisci il tour
        tour_coords = [depot]
        total_prize = 0
        visited_idx = []
        
        for step_idx in tour:
            if step_idx < len(loc):
                tour_coords.append(loc[step_idx])
                total_prize += prize[step_idx]
                visited_idx.append(step_idx)
        
        tour_coords.append(depot)
        tour_coords = np.array(tour_coords)
        
        # Plotta i nodi visitati
        if len(visited_idx) > 0:
            visited_locs = loc[visited_idx]
            visited_prizes = prize[visited_idx]
            scatter = ax.scatter(visited_locs[:, 0], visited_locs[:, 1], 
                               c=visited_prizes, s=200, cmap='YlOrRd',
                               label='Visitati', zorder=3, 
                               edgecolors='black', linewidth=1.5)
            plt.colorbar(scatter, ax=ax, label='Premio')
        
        # Plotta il percorso
        if len(tour_coords) > 1:
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], 
                   'b-', linewidth=2, alpha=0.6, zorder=2)
            
            # Frecce direzione
            for i in range(len(tour_coords) - 1):
                dx = tour_coords[i+1, 0] - tour_coords[i, 0]
                dy = tour_coords[i+1, 1] - tour_coords[i, 1]
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    ax.arrow(tour_coords[i, 0], tour_coords[i, 1], 
                            dx*0.8, dy*0.8, 
                            head_width=0.02, head_length=0.02, 
                            fc='blue', ec='blue', alpha=0.5, zorder=2)
        
        # Calcola lunghezza
        tour_length = 0
        if len(tour_coords) > 1:
            tour_length = np.sum(np.sqrt(np.sum(np.diff(tour_coords, axis=0)**2, axis=1)))
        
        # Titolo
        ax.set_title(f'Istanza {idx+1}\n'
                    f'Premio: {total_prize:.3f} | '
                    f'Lung: {tour_length:.3f}/{max_length:.3f} | '
                    f'Costo: {-cost[idx].item():.3f}\n'
                    f'Visitati: {len(visited_idx)}/{len(loc)}',
                    fontsize=11, fontweight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('op_solutions.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualizzazione salvata in: op_solutions.png")
    print(f"📂 Apri il file con: explorer.exe op_solutions.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path al modello .pt')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--graph_size', type=int, default=20)
    
    args = parser.parse_args()
    plot_op_solution(args.model, args.n_samples, args.graph_size)