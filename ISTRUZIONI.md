# Guida Completa: Attention, Learn to Solve Routing Problems!

## Indice
1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Problemi Supportati](#problemi-supportati)
4. [Generazione Dati](#generazione-dati)
5. [Training dei Modelli](#training-dei-modelli)
6. [Valutazione dei Modelli](#valutazione-dei-modelli)
7. [Opzioni Avanzate](#opzioni-avanzate)
8. [Esempi Pratici Completi](#esempi-pratici-completi)
9. [Risoluzione Problemi](#risoluzione-problemi)

---

## Introduzione

Questo codice implementa un modello basato su **Attention** per risolvere problemi di routing combinatorio usando **Reinforcement Learning**. Il modello è addestrato con l'algoritmo REINFORCE.

### Problemi Risolvibili
- **TSP** (Travelling Salesman Problem): Trova il percorso più breve che visita tutte le città
- **VRP** (Vehicle Routing Problem): Routing con vincoli di capacità del veicolo
- **OP** (Orienteering Problem): Massimizza i premi raccolti entro un limite di distanza
- **PCTSP** (Prize Collecting TSP): Bilanciamento tra lunghezza del tour e premi raccolti

---

## Installazione

### Passo 1: Prerequisiti
Assicurati di avere Python >= 3.8 installato.

### Passo 2: Creazione Ambiente Conda (Opzionale ma Raccomandato)
```bash
# Crea l'ambiente usando il file fornito
conda env create --file environment.yml

# Attiva l'ambiente
conda activate attention_tsp
```

### Passo 3: Installazione Manuale delle Dipendenze
Se preferisci non usare Conda:
```bash
# Installa PyTorch (vedi https://pytorch.org/ per la versione corretta)
pip install torch torchvision

# Installa le altre dipendenze
pip install numpy scipy tqdm tensorboard_logger matplotlib
```

### Passo 4: Verifica Installazione
```bash
# Verifica che PyTorch funzioni
python -c "import torch; print(torch.__version__); print('CUDA disponibile:', torch.cuda.is_available())"
```

---

## Problemi Supportati

### 1. TSP (Travelling Salesman Problem)
**Descrizione**: Trova il percorso più breve che visita tutte le città esattamente una volta e ritorna alla città di partenza.

**Rappresentazione**:
- Input: Coordinate (x, y) di N città in uno spazio 2D [0, 1] × [0, 1]
- Output: Sequenza di città che minimizza la distanza totale

### 2. VRP (Vehicle Routing Problem)
**Descrizione**: Routing di veicoli con capacità limitata che devono servire clienti con domande specifiche.

**Rappresentazione**:
- Deposito: Punto di partenza e arrivo
- Clienti: Coordinate + domanda (1-9)
- Capacità veicolo: Varia con la dimensione del problema
  - 10 nodi: capacità 20
  - 20 nodi: capacità 30
  - 50 nodi: capacità 40
  - 100 nodi: capacità 50

### 3. OP (Orienteering Problem)
**Descrizione**: Massimizza i premi raccolti visitando i nodi, rispettando un limite massimo di distanza.

**Rappresentazione**:
- Deposito: Punto di partenza e arrivo
- Nodi: Coordinate + premio
- Lunghezza massima tour:
  - 20 nodi: 2.0
  - 50 nodi: 3.0
  - 100 nodi: 4.0

**Tipi di premi**:
- `const`: Premi costanti (sempre 1)
- `unif`: Premi uniformi casuali tra 0.01 e 1.00
- `dist`: Premi proporzionali alla distanza dal deposito

### 4. PCTSP (Prize Collecting TSP)
**Descrizione**: Bilanciamento tra lunghezza del tour e premi raccolti. Devi visitare abbastanza nodi per raggiungere un premio minimo.

**Rappresentazione**:
- Deposito: Punto di partenza
- Nodi: Coordinate + premio + penalità (se non visitato)
- Vincolo: Somma dei premi raccolti >= 1.0

---

## Generazione Dati

### Esempio 1: Generare Dati di Validazione per Tutti i Problemi

```bash
# Genera 10000 istanze per validazione con seed 4321
python generate_data.py --problem all --name validation --seed 4321 --dataset_size 10000
```

**Cosa succede**:
- Crea la cartella `data/` se non esiste
- Genera sottocartelle per ogni problema: `data/tsp/`, `data/vrp/`, `data/op/`, `data/pctsp/`
- Crea file `.pkl` per ogni combinazione di problema e dimensione:
  - `tsp20_validation_seed4321.pkl`
  - `tsp50_validation_seed4321.pkl`
  - `tsp100_validation_seed4321.pkl`
  - `vrp20_validation_seed4321.pkl`
  - ... e così via

### Esempio 2: Generare Solo Dati TSP per Test

```bash
# Genera dati di test solo per TSP
python generate_data.py --problem tsp --name test --seed 1234 --dataset_size 10000
```

**Output**:
```
data/tsp/tsp20_test_seed1234.pkl
data/tsp/tsp50_test_seed1234.pkl
data/tsp/tsp100_test_seed1234.pkl
```

### Esempio 3: Generare Dati per una Dimensione Specifica

```bash
# Genera solo TSP con 20 nodi
python generate_data.py --problem tsp --name mio_dataset --seed 5678 --graph_sizes 20
```

### Esempio 4: Generare Orienteering Problem con Diversi Tipi di Premi

```bash
# Solo OP con premi uniformi
python generate_data.py --problem op --data_distribution unif --name test --seed 1234

# Solo OP con premi basati su distanza
python generate_data.py --problem op --data_distribution dist --name test --seed 1234
```

### Parametri Importanti per generate_data.py

| Parametro | Descrizione | Default | Esempio |
|-----------|-------------|---------|---------|
| `--problem` | Tipo di problema: `tsp`, `vrp`, `op`, `pctsp`, o `all` | `all` | `--problem tsp` |
| `--name` | Nome identificativo del dataset | **Obbligatorio** | `--name validation` |
| `--seed` | Seed per riproducibilità | `1234` | `--seed 4321` |
| `--dataset_size` | Numero di istanze da generare | `10000` | `--dataset_size 5000` |
| `--graph_sizes` | Dimensioni dei grafi da generare | `[20, 50, 100]` | `--graph_sizes 20 50` |
| `--data_distribution` | Distribuzione dati (per OP) | `all` | `--data_distribution unif` |

---

## Training dei Modelli

### Esempio 1: Training Base TSP con 20 Nodi

```bash
# Training semplice senza baseline
python run.py --problem tsp --graph_size 20 --run_name 'mio_primo_tsp'
```

**Cosa succede**:
1. Il modello viene creato con le impostazioni di default
2. Vengono generati dati di training al volo (1,280,000 istanze per epoch)
3. Il training dura 100 epoch (default)
4. I modelli vengono salvati in `outputs/tsp_20/mio_primo_tsp_YYYYMMDDTHHMMSS/`
5. I log TensorBoard vengono salvati in `logs/`

**Output previsto**:
```
Epoch 0: avg_reward: -3.85
Epoch 1: avg_reward: -3.72
...
```

### Esempio 2: Training TSP con Rollout Baseline (Raccomandato)

```bash
# Training con rollout baseline e dataset di validazione
python run.py \
  --problem tsp \
  --graph_size 20 \
  --baseline rollout \
  --run_name 'tsp20_rollout' \
  --val_dataset data/tsp/tsp20_validation_seed4321.pkl
```

**Vantaggi del rollout baseline**:
- Training più stabile
- Convergenza più veloce
- Risultati migliori

**Spiegazione parametri**:
- `--baseline rollout`: Usa una politica greedy come baseline per ridurre la varianza
- `--val_dataset`: Usa un dataset fisso per validazione (risultati confrontabili)

### Esempio 3: Training VRP (Vehicle Routing Problem)

```bash
# Training VRP con 50 nodi
python run.py \
  --problem vrp \
  --graph_size 50 \
  --baseline rollout \
  --run_name 'vrp50_rollout' \
  --val_dataset data/vrp/vrp50_validation_seed4321.pkl \
  --n_epochs 100
```

**Nota**: VRP richiede più tempo di training rispetto a TSP per la maggiore complessità.

### Esempio 4: Training con Multiple GPU

```bash
# Usa solo le GPU 0 e 1
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --problem tsp \
  --graph_size 100 \
  --baseline rollout \
  --run_name 'tsp100_multigpu'

# Disabilita completamente CUDA (solo CPU)
python run.py --problem tsp --graph_size 20 --no_cuda
```

### Esempio 5: Training con Batch Size Personalizzato

```bash
# Riduce batch size per risparmiare memoria
python run.py \
  --problem tsp \
  --graph_size 100 \
  --baseline rollout \
  --batch_size 256 \
  --eval_batch_size 512 \
  --run_name 'tsp100_small_batch'
```

### Esempio 6: Fine-tuning da Modello Pre-addestrato

```bash
# Carica un modello pre-addestrato e continua il training
python run.py \
  --problem tsp \
  --graph_size 100 \
  --baseline rollout \
  --load_path pretrained/tsp_100/epoch-99.pt \
  --run_name 'tsp100_finetuned' \
  --n_epochs 20
```

**Nota**: `--load_path` carica solo i parametri del modello, non l'ottimizzatore.

### Esempio 7: Riprendere Training Interrotto

```bash
# Riprende esattamente da dove si era interrotto
python run.py \
  --resume outputs/tsp_20/tsp20_rollout_20231115T143022/epoch-10.pt
```

**Cosa viene ripristinato con --resume**:
- Parametri del modello
- Stato dell'ottimizzatore
- Stato della baseline
- Epoch corrente
- Stato del generatore di numeri casuali

### Esempio 8: Training Orienteering Problem

```bash
# Training OP con premi uniformi
python run.py \
  --problem op \
  --graph_size 20 \
  --baseline rollout \
  --data_distribution unif \
  --run_name 'op20_unif'
```

### Esempio 9: Training PCTSP

```bash
# Training Prize Collecting TSP
python run.py \
  --problem pctsp \
  --graph_size 20 \
  --baseline rollout \
  --run_name 'pctsp20_rollout'
```

### Parametri Importanti per Training

#### Dati
| Parametro | Descrizione | Default | Esempio |
|-----------|-------------|---------|---------|
| `--problem` | Problema da risolvere | `tsp` | `--problem vrp` |
| `--graph_size` | Numero di nodi | `20` | `--graph_size 100` |
| `--batch_size` | Istanze per batch | `512` | `--batch_size 256` |
| `--epoch_size` | Istanze per epoch | `1280000` | `--epoch_size 640000` |
| `--val_dataset` | Dataset di validazione | `None` | `--val_dataset data/tsp/...` |

#### Modello
| Parametro | Descrizione | Default | Esempio |
|-----------|-------------|---------|---------|
| `--model` | Tipo di modello | `attention` | `--model pointer` |
| `--embedding_dim` | Dimensione embedding | `128` | `--embedding_dim 256` |
| `--hidden_dim` | Dimensione layer nascosti | `128` | `--hidden_dim 256` |
| `--n_encode_layers` | Strati nell'encoder | `3` | `--n_encode_layers 6` |

#### Training
| Parametro | Descrizione | Default | Esempio |
|-----------|-------------|---------|---------|
| `--baseline` | Tipo di baseline | `None` | `--baseline rollout` |
| `--n_epochs` | Numero di epoch | `100` | `--n_epochs 200` |
| `--lr_model` | Learning rate attore | `1e-4` | `--lr_model 5e-5` |
| `--lr_critic` | Learning rate critico | `1e-4` | `--lr_critic 5e-5` |
| `--lr_decay` | Decay learning rate | `1.0` | `--lr_decay 0.99` |
| `--no_cuda` | Disabilita GPU | `False` | `--no_cuda` |
| `--seed` | Seed casuale | `1234` | `--seed 5678` |

#### Baseline
| Parametro | Descrizione | Default | Esempio |
|-----------|-------------|---------|---------|
| `--baseline` | `rollout`, `critic`, `exponential` | `None` | `--baseline rollout` |
| `--bl_warmup_epochs` | Epoch di warmup | `1` (rollout) | `--bl_warmup_epochs 5` |
| `--bl_alpha` | Alpha per test statistico | `0.05` | `--bl_alpha 0.1` |

#### Output
| Parametro | Descrizione | Default | Esempio |
|-----------|-------------|---------|---------|
| `--run_name` | Nome del run | `run` | `--run_name esperimento1` |
| `--output_dir` | Directory output | `outputs` | `--output_dir risultati` |
| `--log_dir` | Directory log | `logs` | `--log_dir tensorboard_logs` |
| `--checkpoint_epochs` | Salva ogni N epoch | `1` | `--checkpoint_epochs 5` |

---

## Valutazione dei Modelli

### Esempio 1: Valutazione Base con Strategia Greedy

```bash
# Valuta il modello migliore usando strategia greedy
python eval.py \
  data/tsp/tsp20_test_seed1234.pkl \
  --model pretrained/tsp_20 \
  --decode_strategy greedy
```

**Output**:
```
Evaluate greedy decoding on tsp20_test_seed1234
Average cost: 3.8345
Avg duration: 0.0234s
Saved results to: outputs/tsp_20/...
```

### Esempio 2: Valutazione con Campionamento (Migliori Risultati)

```bash
# Campiona 1280 soluzioni per ogni istanza e prendi la migliore
python eval.py \
  data/tsp/tsp20_test_seed1234.pkl \
  --model pretrained/tsp_20 \
  --decode_strategy sample \
  --width 1280 \
  --eval_batch_size 1
```

**Spiegazione**:
- `--width 1280`: Genera 1280 soluzioni diverse
- `--eval_batch_size 1`: Processa un'istanza alla volta (necessario per sampling)
- Più alto è `--width`, migliori sono i risultati ma più lento è il processo

### Esempio 3: Valutazione con Beam Search

```bash
# Usa beam search con ampiezza 10
python eval.py \
  data/tsp/tsp50_test_seed1234.pkl \
  --model pretrained/tsp_50 \
  --decode_strategy bs \
  --width 10
```

### Esempio 4: Valutare un Epoch Specifico

```bash
# Valuta l'epoch 50 invece dell'ultimo
python eval.py \
  data/tsp/tsp100_test_seed1234.pkl \
  --model outputs/tsp_100/mio_run_20231115T143022/epoch-50.pt \
  --decode_strategy greedy
```

### Esempio 5: Valutazione su Multiple Dataset

```bash
# Valuta su più dataset contemporaneamente
for size in 20 50 100; do
  python eval.py \
    data/tsp/tsp${size}_test_seed1234.pkl \
    --model pretrained/tsp_${size} \
    --decode_strategy greedy
done
```

### Esempio 6: Valutazione VRP

```bash
# Valuta VRP con 100 nodi
python eval.py \
  data/vrp/vrp100_test_seed1234.pkl \
  --model pretrained/vrp_100 \
  --decode_strategy greedy
```

### Esempio 7: Valutazione Durante il Training

```bash
# Valuta senza salvare risultati (più veloce)
python run.py \
  --problem tsp \
  --graph_size 20 \
  --eval_only \
  --load_path outputs/tsp_20/tsp20_rollout_20231115T143022/epoch-99.pt \
  --val_dataset data/tsp/tsp20_test_seed1234.pkl
```

### Strategie di Decodifica Spiegate

#### 1. Greedy (Veloce)
```bash
--decode_strategy greedy
```
- Ad ogni passo, seleziona il nodo con probabilità più alta
- **Veloce**: 0.02-0.05s per istanza
- **Qualità**: Buona ma non ottima

#### 2. Sampling (Migliore Qualità)
```bash
--decode_strategy sample --width 1280
```
- Genera molte soluzioni diverse campionando dalla distribuzione
- Seleziona la migliore
- **Lento**: ~10-30s per istanza (dipende da width)
- **Qualità**: Eccellente

#### 3. Beam Search (Bilanciato)
```bash
--decode_strategy bs --width 10
```
- Mantiene le top-k soluzioni parziali ad ogni passo
- **Velocità**: Media
- **Qualità**: Tra greedy e sampling

---

## Opzioni Avanzate

### 1. Ottimizzazione Memoria

#### Checkpoint dell'Encoder
```bash
# Riduce l'uso di memoria durante training (utile per grafi grandi)
python run.py \
  --problem tsp \
  --graph_size 100 \
  --baseline rollout \
  --checkpoint_encoder \
  --run_name 'tsp100_lowmem'
```

**Quando usarlo**: Grafi con 100+ nodi e memoria GPU limitata.

#### Shrink Batch Size
```bash
# Riduce dinamicamente il batch quando le istanze sono completate
python run.py \
  --problem vrp \
  --graph_size 100 \
  --baseline rollout \
  --shrink_size 50 \
  --run_name 'vrp100_shrink'
```

**Quando usarlo**: VRP con molti nodi dove alcuni tour finiscono prima.

### 2. Learning Rate Scheduling

```bash
# Decay del learning rate dello 0.96 ogni epoch
python run.py \
  --problem tsp \
  --graph_size 50 \
  --baseline rollout \
  --lr_model 1e-4 \
  --lr_decay 0.96 \
  --n_epochs 100 \
  --run_name 'tsp50_lr_decay'
```

### 3. Baseline Personalizzate

#### Exponential Moving Average Baseline
```bash
# Usa baseline esponenziale (più leggera di rollout)
python run.py \
  --problem tsp \
  --graph_size 20 \
  --baseline exponential \
  --exp_beta 0.8 \
  --run_name 'tsp20_exp'
```

#### Critic Baseline
```bash
# Usa rete critic come baseline
python run.py \
  --problem tsp \
  --graph_size 20 \
  --baseline critic \
  --lr_critic 1e-4 \
  --run_name 'tsp20_critic'
```

#### Rollout con Warmup Personalizzato
```bash
# 10 epoch di warmup con exponential prima di passare a rollout
python run.py \
  --problem tsp \
  --graph_size 50 \
  --baseline rollout \
  --bl_warmup_epochs 10 \
  --run_name 'tsp50_warmup10'
```

### 4. Modello Personalizzato

```bash
# Modello più grande per problemi complessi
python run.py \
  --problem vrp \
  --graph_size 100 \
  --baseline rollout \
  --embedding_dim 256 \
  --hidden_dim 256 \
  --n_encode_layers 6 \
  --run_name 'vrp100_large'
```

**Nota**: Modelli più grandi richiedono più memoria e tempo ma possono dare risultati migliori.

### 5. Visualizzazione con TensorBoard

```bash
# Durante o dopo il training, visualizza i progressi
tensorboard --logdir=logs

# Se vuoi specificare una porta diversa
tensorboard --logdir=logs --port=6007
```

Apri il browser su `http://localhost:6006` (o la porta specificata).

### 6. Disabilitare Logging

```bash
# Disabilita TensorBoard (più veloce)
python run.py \
  --problem tsp \
  --graph_size 20 \
  --no_tensorboard \
  --no_progress_bar \
  --run_name 'tsp20_nolog'
```

---

## Esempi Pratici Completi

### Progetto Completo 1: Training e Valutazione TSP

#### Passo 1: Genera i Dataset
```bash
# Crea directory per i dati
mkdir -p data/tsp

# Genera dati di validazione
python generate_data.py \
  --problem tsp \
  --name validation \
  --seed 4321 \
  --dataset_size 10000 \
  --graph_sizes 20 50 100

# Genera dati di test
python generate_data.py \
  --problem tsp \
  --name test \
  --seed 1234 \
  --dataset_size 10000 \
  --graph_sizes 20 50 100
```

#### Passo 2: Training Modelli
```bash
# Training TSP-20
python run.py \
  --problem tsp \
  --graph_size 20 \
  --baseline rollout \
  --run_name 'tsp20_final' \
  --val_dataset data/tsp/tsp20_validation_seed4321.pkl \
  --n_epochs 100 \
  --checkpoint_epochs 10

# Training TSP-50
python run.py \
  --problem tsp \
  --graph_size 50 \
  --baseline rollout \
  --run_name 'tsp50_final' \
  --val_dataset data/tsp/tsp50_validation_seed4321.pkl \
  --n_epochs 100 \
  --checkpoint_epochs 10

# Training TSP-100
python run.py \
  --problem tsp \
  --graph_size 100 \
  --baseline rollout \
  --run_name 'tsp100_final' \
  --val_dataset data/tsp/tsp100_validation_seed4321.pkl \
  --n_epochs 100 \
  --checkpoint_epochs 10 \
  --checkpoint_encoder
```

#### Passo 3: Valutazione con Greedy
```bash
# Valuta tutti i modelli con greedy
for size in 20 50 100; do
  echo "Evaluating TSP-${size}..."
  python eval.py \
    data/tsp/tsp${size}_test_seed1234.pkl \
    --model outputs/tsp_${size}/tsp${size}_final_*/epoch-99.pt \
    --decode_strategy greedy
done
```

#### Passo 4: Valutazione con Sampling
```bash
# Valuta con sampling per risultati migliori
for size in 20 50 100; do
  echo "Evaluating TSP-${size} with sampling..."
  python eval.py \
    data/tsp/tsp${size}_test_seed1234.pkl \
    --model outputs/tsp_${size}/tsp${size}_final_*/epoch-99.pt \
    --decode_strategy sample \
    --width 1280 \
    --eval_batch_size 1
done
```

### Progetto Completo 2: Transfer Learning TSP

#### Scenario: Addestrare TSP-20, poi fare fine-tuning su TSP-50

```bash
# Passo 1: Training base TSP-20
python run.py \
  --problem tsp \
  --graph_size 20 \
  --baseline rollout \
  --run_name 'tsp20_base' \
  --n_epochs 100

# Passo 2: Fine-tuning su TSP-50 partendo da TSP-20
python run.py \
  --problem tsp \
  --graph_size 50 \
  --baseline rollout \
  --load_path outputs/tsp_20/tsp20_base_*/epoch-99.pt \
  --run_name 'tsp50_transfer' \
  --n_epochs 50 \
  --lr_model 5e-5

# Passo 3: Confronta con training da zero
python run.py \
  --problem tsp \
  --graph_size 50 \
  --baseline rollout \
  --run_name 'tsp50_scratch' \
  --n_epochs 100
```

### Progetto Completo 3: VRP con Diverse Dimensioni

```bash
# Genera dati VRP
python generate_data.py \
  --problem vrp \
  --name validation \
  --seed 4321 \
  --graph_sizes 20 50 100

python generate_data.py \
  --problem vrp \
  --name test \
  --seed 1234 \
  --graph_sizes 20 50 100

# Training VRP-20 (veloce)
python run.py \
  --problem vrp \
  --graph_size 20 \
  --baseline rollout \
  --run_name 'vrp20' \
  --val_dataset data/vrp/vrp20_validation_seed4321.pkl \
  --n_epochs 100

# Training VRP-50 (medio)
python run.py \
  --problem vrp \
  --graph_size 50 \
  --baseline rollout \
  --run_name 'vrp50' \
  --val_dataset data/vrp/vrp50_validation_seed4321.pkl \
  --n_epochs 100 \
  --batch_size 256

# Training VRP-100 (lento, richiede molta memoria)
python run.py \
  --problem vrp \
  --graph_size 100 \
  --baseline rollout \
  --run_name 'vrp100' \
  --val_dataset data/vrp/vrp100_validation_seed4321.pkl \
  --n_epochs 100 \
  --batch_size 128 \
  --checkpoint_encoder \
  --shrink_size 50
```

### Progetto Completo 4: Orienteering Problem con Tutti i Tipi di Premi

```bash
# Genera dati per tutti i tipi di distribuzione
for dist in const unif dist; do
  python generate_data.py \
    --problem op \
    --data_distribution ${dist} \
    --name validation \
    --seed 4321 \
    --graph_sizes 20

  python generate_data.py \
    --problem op \
    --data_distribution ${dist} \
    --name test \
    --seed 1234 \
    --graph_sizes 20
done

# Training per ogni tipo
for dist in const unif dist; do
  python run.py \
    --problem op \
    --graph_size 20 \
    --data_distribution ${dist} \
    --baseline rollout \
    --run_name "op20_${dist}" \
    --val_dataset data/op/op_${dist}20_validation_seed4321.pkl \
    --n_epochs 100
done

# Valutazione comparativa
for dist in const unif dist; do
  echo "Evaluating OP ${dist}..."
  python eval.py \
    data/op/op_${dist}20_test_seed1234.pkl \
    --model outputs/op_20/op20_${dist}_*/epoch-99.pt \
    --decode_strategy greedy
done
```

### Progetto Completo 5: Esperimenti con Diverse Baseline

```bash
# Stesso problema, diverse baseline per confronto
PROBLEM="tsp"
SIZE=20

# Nessuna baseline
python run.py \
  --problem ${PROBLEM} \
  --graph_size ${SIZE} \
  --run_name "${PROBLEM}${SIZE}_no_baseline" \
  --n_epochs 50

# Exponential baseline
python run.py \
  --problem ${PROBLEM} \
  --graph_size ${SIZE} \
  --baseline exponential \
  --run_name "${PROBLEM}${SIZE}_exponential" \
  --n_epochs 50

# Critic baseline
python run.py \
  --problem ${PROBLEM} \
  --graph_size ${SIZE} \
  --baseline critic \
  --run_name "${PROBLEM}${SIZE}_critic" \
  --n_epochs 50

# Rollout baseline
python run.py \
  --problem ${PROBLEM} \
  --graph_size ${SIZE} \
  --baseline rollout \
  --run_name "${PROBLEM}${SIZE}_rollout" \
  --n_epochs 50

# Valuta tutti
for baseline in no_baseline exponential critic rollout; do
  python eval.py \
    data/${PROBLEM}/${PROBLEM}${SIZE}_test_seed1234.pkl \
    --model outputs/${PROBLEM}_${SIZE}/${PROBLEM}${SIZE}_${baseline}_*/epoch-49.pt \
    --decode_strategy greedy
done
```

### Progetto Completo 6: Confronto con Baselines Classiche

```bash
# Prima installa i baselines (vedi sezione Baselines sotto)
cd problems/tsp
bash install_concorde.sh
cd ../..

# Genera dati test
python generate_data.py --problem tsp --name test --seed 1234 --graph_sizes 20 50 100

# Training del modello RL
python run.py \
  --problem tsp \
  --graph_size 50 \
  --baseline rollout \
  --run_name 'tsp50_vs_baseline' \
  --n_epochs 100

# Valuta il modello RL
python eval.py \
  data/tsp/tsp50_test_seed1234.pkl \
  --model outputs/tsp_50/tsp50_vs_baseline_*/epoch-99.pt \
  --decode_strategy greedy

# Valuta baselines classiche
python -m problems.tsp.tsp_baseline farthest_insertion data/tsp/tsp50_test_seed1234.pkl
python -m problems.tsp.tsp_baseline nearest_insertion data/tsp/tsp50_test_seed1234.pkl
python -m problems.tsp.tsp_baseline concorde data/tsp/tsp50_test_seed1234.pkl
```

---

## Baselines Classiche

Il repository include implementazioni di algoritmi classici per confronto.

### TSP Baselines

#### Installazione Concorde (Solver Esatto)
```bash
cd problems/tsp
bash install_concorde.sh
cd ../..
```

#### Esecuzione Baselines TSP
```bash
# Farthest Insertion
python -m problems.tsp.tsp_baseline farthest_insertion \
  data/tsp/tsp20_test_seed1234.pkl \
  data/tsp/tsp50_test_seed1234.pkl

# Nearest Insertion
python -m problems.tsp.tsp_baseline nearest_insertion \
  data/tsp/tsp20_test_seed1234.pkl

# Concorde (solver esatto)
python -m problems.tsp.tsp_baseline concorde \
  data/tsp/tsp20_test_seed1234.pkl
```

**Baselines disponibili per TSP**:
- `farthest_insertion`: Euristica di inserimento
- `nearest_insertion`: Euristica di inserimento
- `random`: Soluzione casuale
- `concorde`: Solver esatto (lento ma ottimo)

### VRP Baselines

```bash
# Richiede installazione di LKH3 (automatica al primo uso)
python -m problems.vrp.vrp_baseline lkh \
  data/vrp/vrp20_test_seed1234.pkl
```

### OP (Orienteering) Baselines

#### Installazione Compass
```bash
cd problems/op
bash install_compass.sh
cd ../..
```

#### Esecuzione
```bash
# Compass baseline per OP
python -m problems.op.op_baseline compass \
  data/op/op_const20_test_seed1234.pkl
```

### PCTSP Baselines

```bash
# Baseline per PCTSP
python -m problems.pctsp.pctsp_baseline \
  data/pctsp/pctsp20_test_seed1234.pkl
```

---

## Visualizzazione Risultati

### Visualizzare una Soluzione VRP (Notebook)

Il repository include `plot_vrp.ipynb` per visualizzare soluzioni VRP.

```python
# In Jupyter Notebook
import torch
import matplotlib.pyplot as plt
from nets.attention_model import AttentionModel
from problems.vrp.problem_vrp import CVRP

# Carica il modello
model = AttentionModel.load('pretrained/vrp_100/epoch-99.pt')

# Genera un'istanza VRP
problem = CVRP
dataset = problem.make_dataset(size=100, num_samples=1)

# Risolvi
model.eval()
with torch.no_grad():
    tour, _ = model(dataset)

# Visualizza (vedi notebook per codice completo)
```

### Creare Grafici di Training

```python
# Script per estrarre metriche dai log
import tensorboard
# ... usa TensorBoard per estrarre curve di training
```

---

## Risoluzione Problemi

### Problema 1: Out of Memory (OOM)

**Sintomo**: `RuntimeError: CUDA out of memory`

**Soluzioni**:
```bash
# 1. Riduci batch size
python run.py ... --batch_size 128

# 2. Usa checkpoint encoder
python run.py ... --checkpoint_encoder

# 3. Riduci dimensione modello
python run.py ... --embedding_dim 64 --hidden_dim 64

# 4. Usa shrink_size
python run.py ... --shrink_size 32

# 5. Usa meno GPU o solo CPU
CUDA_VISIBLE_DEVICES=0 python run.py ...
python run.py ... --no_cuda
```

### Problema 2: Training Molto Lento

**Soluzioni**:
```bash
# 1. Riduci epoch_size
python run.py ... --epoch_size 640000

# 2. Usa exponential baseline invece di rollout
python run.py ... --baseline exponential

# 3. Aumenta eval_batch_size
python run.py ... --eval_batch_size 2048

# 4. Disabilita logging non necessario
python run.py ... --no_tensorboard --no_progress_bar
```

### Problema 3: Risultati Non Migliorano

**Soluzioni**:
```bash
# 1. Assicurati di usare una baseline
python run.py ... --baseline rollout

# 2. Prova learning rate diverso
python run.py ... --lr_model 5e-5

# 3. Aumenta numero di epoch
python run.py ... --n_epochs 200

# 4. Usa modello più grande
python run.py ... --embedding_dim 256 --n_encode_layers 6
```

### Problema 4: Validazione Diversa da Test

**Causa**: Seed o dataset diversi

**Soluzione**:
```bash
# Assicurati di usare seed diversi per validation e test
python generate_data.py --name validation --seed 4321
python generate_data.py --name test --seed 1234

# E di valutare sul dataset corretto
python eval.py data/tsp/tsp20_test_seed1234.pkl ...
```

### Problema 5: Errore nell'Import di Moduli

**Sintomo**: `ModuleNotFoundError: No module named 'tensorboard_logger'`

**Soluzione**:
```bash
pip install tensorboard_logger numpy scipy tqdm matplotlib
```

### Problema 6: CUDA Non Disponibile

**Sintomo**: `torch.cuda.is_available() returns False`

**Verifica**:
```bash
# Verifica installazione CUDA
nvidia-smi

# Reinstalla PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problema 7: Concorde Non Si Installa

**Soluzione**:
```bash
# Dipendenze necessarie per Concorde
sudo apt-get install build-essential libgmp3-dev

# Poi riprova installazione
cd problems/tsp
bash install_concorde.sh
```

---

## Comandi Utili Rapidi

### Quick Start - Training Veloce
```bash
# TSP-20 in 30 minuti circa
python run.py --graph_size 20 --baseline rollout --run_name 'quick_test' --n_epochs 10
```

### Visualizzare Aiuto
```bash
# Help completo per training
python run.py -h

# Help completo per valutazione
python eval.py -h

# Help per generazione dati
python generate_data.py -h
```

### Visualizzare Modelli Salvati
```bash
# Lista tutti i modelli salvati
find outputs/ -name "*.pt" -type f

# Trova i modelli più recenti
find outputs/ -name "epoch-*.pt" -type f -printf '%T@ %p\n' | sort -n | tail -5
```

### Cleanup
```bash
# Rimuovi tutti i log TensorBoard
rm -rf logs/

# Rimuovi modelli vecchi mantenendo solo gli ultimi epoch
# (fai attenzione con questo comando!)
find outputs/ -name "epoch-[0-9].pt" -type f -delete  # Rimuove epoch 0-9
```

---

## Riassunto Flusso di Lavoro Tipico

1. **Setup Iniziale**
   ```bash
   conda env create --file environment.yml
   conda activate attention_tsp
   ```

2. **Genera Dati**
   ```bash
   python generate_data.py --problem all --name validation --seed 4321
   python generate_data.py --problem all --name test --seed 1234
   ```

3. **Training**
   ```bash
   python run.py --problem tsp --graph_size 20 --baseline rollout \
     --run_name 'tsp20' --val_dataset data/tsp/tsp20_validation_seed4321.pkl
   ```

4. **Valutazione**
   ```bash
   python eval.py data/tsp/tsp20_test_seed1234.pkl \
     --model outputs/tsp_20/tsp20_*/epoch-99.pt --decode_strategy greedy
   ```

5. **Ottimizzazione** (Opzionale)
   ```bash
   python eval.py data/tsp/tsp20_test_seed1234.pkl \
     --model outputs/tsp_20/tsp20_*/epoch-99.pt \
     --decode_strategy sample --width 1280 --eval_batch_size 1
   ```

---

## Riferimenti e Risorse

### Paper Originale
```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```

### Repository Correlati
- Implementazione più recente: https://github.com/ai4co/rl4co
- Altra implementazione: https://github.com/cpwan/RLOR

### Link Utili
- PyTorch: https://pytorch.org/
- TensorBoard: https://www.tensorflow.org/tensorboard
- Concorde TSP Solver: http://www.math.uwaterloo.ca/tsp/concorde.html
- LKH3: http://akira.ruc.dk/~keld/research/LKH-3/

---

## Conclusione

Questa guida copre tutte le funzionalità principali del codice. Per domande specifiche o problemi non coperti, consulta il codice sorgente o apri una issue sul repository GitHub.

Buon lavoro con i problemi di routing! 🚀
