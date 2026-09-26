# AI Model Lab V4 Ultimate - Manuale completo

**Versione:** 4.0 Ultimate Dual Engine  
**Scopo:** comprendere, costruire, addestrare e ispezionare modelli neurali da un singolo neurone fino a un Transformer decoder PyTorch moderno.

> **Principio del progetto:** nessun valore importante deve cambiare senza che l’utente possa osservare cosa lo ha causato. La modalità Didattica massimizza la trasparenza; Framework Pro V4 massimizza la fedeltà a un vero stack PyTorch locale.

## 1. Cosa contiene V4

V4 mantiene tutte le funzioni V2/V3 e aggiunge un secondo livello professionale: Modern Transformer con RMSNorm, Rotary Positional Embeddings (RoPE), SwiGLU, Grouped Query Attention (GQA), PyTorch Scaled Dot Product Attention, LoRA, gradient accumulation, cosine learning-rate schedule con warmup, gradient checkpointing, mixed precision, training in background, tokenizer Byte-BPE addestrabile, checkpoint `.pt`, tracciamento esperimenti SQLite, embedding microscope e Scale & Memory Lab.

Il progetto resta locale. Non sostiene che un piccolo modello addestrato sul PC abbia capacità paragonabili a grandi modelli industriali addestrati su cluster: l’architettura e i meccanismi sono reali; la scala dipende dall’hardware, dai dati e dal tempo di training.

---

AI Model Lab V4 è un laboratorio didattico interattivo progettato per rendere visibili i meccanismi interni di un modello neurale: input, pesi, bias, attivazioni, forward pass, loss, gradienti, backpropagation, update dei parametri, generalizzazione e diagnostica del training.

Il progetto non vuole nascondere il calcolo dietro una libreria. Il motore MLP principale implementa esplicitamente i layer Dense, le funzioni di attivazione, la loss, la propagazione del gradiente e gli optimizer SGD/Adam. La sezione Mini Transformer è invece una visualizzazione didattica della self-attention e non deve essere interpretata come un LLM completo.

---

## 2. Come leggere l'interfaccia globale

### Barra laterale
La barra laterale contiene le sezioni dell'app. Ogni voce cambia la sezione centrale senza ricaricare la pagina.

- **Dashboard**: controllo principale di training e visualizzazione live.
- **Neuron Lab**: analisi matematica di un singolo neurone.
- **Network Builder**: struttura della rete e numero di neuroni.
- **Teacher vs Student**: Agente A pone esempi, Agente B prova a predire e si corregge.
- **Train My Data**: inserimento manuale di coppie input/risposta.
- **Forward / Backprop**: avanzamento didattico per fase.
- **Parameters**: ispezione di ogni peso e bias.
- **Dataset Surface**: interpretazione del dataset e decision surface.
- **Computational Graph**: valori intermedi z e attivazioni a.
- **Model X-Ray**: riepilogo strutturale del modello.
- **Generalization**: train/validation/test e overfitting.
- **Diagnostics**: salute dei gradienti e dei neuroni.
- **Optimizer Arena**: confronto controllato SGD vs Adam.
- **Mini Transformer**: tokenizzazione didattica e self-attention.
- **Documentation**: legenda rapida integrata.

### Pulsanti permanenti della barra laterale
- **Python Live**: apre una pagina separata sincronizzata con il modello. La pagina può anche usare Pyodide per eseguire Python reale nel browser.
- **Export Model**: salva in JSON architettura, activation, pesi e bias. Serve per conservare o confrontare uno stato addestrato.
- **Import Model**: carica un JSON esportato. Sostituisce il modello corrente con quello contenuto nel file.
- **Export Dataset**: salva gli esempi correnti in JSON.

### Header
- **Tour guidato**: evidenzia progressivamente i controlli fondamentali e spiega cosa osservare.
- **Help contestuale**: abilita un comportamento didattico e ricorda che i controlli principali possiedono tooltip `?`.
- **Stato**: indica se il training è pronto, attivo o in pausa.
- **Engine: MLP reale**: ricorda che la rete principale è un multilayer perceptron effettivamente calcolato, non una sola animazione grafica.

### Metriche globali
- **Epoca**: numero di cicli visualizzati completati. Un'epoca nell'app corrisponde a un batch visualizzato, non necessariamente a una scansione completa di tutto il dataset.
- **Loss**: errore numerico dell'ultimo batch/esempio elaborato. Più è bassa, migliore è il fit sul dato corrente, ma una loss bassa da sola non garantisce generalizzazione.
- **Parametri**: numero totale di pesi e bias trainabili.
- **Architettura**: dimensione di input, hidden layer e output, per esempio `1→4→4→1`.
- **Learning Rate**: ampiezza dell'aggiornamento dei parametri.
- **Fase**: `idle`, `forward`, `loss`, `backprop` o `update`.

---

## 3. Dashboard - ogni controllo

### Task
**Label: `Task`**

- **Regressione**: la rete produce un valore numerico continuo. Il layer di output usa activation lineare e la loss principale è MSE semplificata `0.5*(prediction-target)^2`.
- **Classificazione binaria**: l'output usa sigmoid e rappresenta una probabilità tra 0 e 1. La loss è Binary Cross Entropy.

Cambiare Task porta a usare un dataset compatibile e ricostruire la rete.

### Dataset
**Label: `Dataset`**

- **Linear y=2.5x+1**: regressione quasi lineare con piccolo rumore.
- **Sine**: regressione non lineare `sin(x)`, più interessante per osservare il ruolo degli hidden layer.
- **XOR**: classificazione non linearmente separabile; dimostra perché un singolo neurone lineare non basta.
- **Circles**: classificazione con frontiera circolare; rende visibile la decision surface.

### Hidden layer sizes
Campo testuale come `4,4` o `8,6,4`.

- `4,4` = due hidden layer da 4 neuroni.
- `8,6,4` = tre hidden layer da 8, 6 e 4 neuroni.

Aumentare i neuroni aumenta il numero di parametri e quindi la capacità del modello, ma può anche aumentare costo computazionale e rischio di overfitting.

Per un Dense layer con `n_input` ingressi e `n_output` neuroni:

`parametri = n_input*n_output + n_output`

Il secondo termine sono i bias.

### Activation
- **tanh**: output tra -1 e 1; utile didatticamente, ma può saturare.
- **relu**: `max(0,z)`; semplice e molto usata, ma alcuni neuroni possono diventare "morti".
- **sigmoid**: output tra 0 e 1; può saturare e produrre gradienti piccoli.

### Optimizer
- **SGD**: `p <- p - learning_rate * gradient`. È il modo più diretto per capire il gradient descent.
- **Adam**: mantiene medie mobili del gradiente e del gradiente quadratico; il passo effettivo è adattato per parametro.

### Seed riproducibile
Un intero usato dal generatore pseudocasuale.

**Pulsante `Applica`**: reimposta la sequenza casuale e ricrea dataset e pesi. Se seed, architettura e impostazioni sono uguali, puoi ripetere lo stesso esperimento.

### Preset Learning Rate
Pulsanti `0.001`, `0.01`, `0.03`, `0.1`.

- `0.001`: lento ma spesso stabile.
- `0.01`: moderato.
- `0.03`: default didattico.
- `0.1`: aggressivo; utile per osservare instabilità su alcuni problemi.

### Learning Rate slider
Controlla `η` nella formula di update.

`nuovo_parametro = vecchio_parametro - η * gradiente`

Troppo piccolo: training lento. Troppo grande: oscillazioni, divergenza o gradienti numericamente problematici.

### Velocità
Ritardo in millisecondi tra un update visualizzato e il successivo. Cambia soltanto l'esperienza visiva, non la formula matematica.

### Batch size
Numero di esempi estratti dal training split per ogni epoca visualizzata. Un batch più grande rende la loss visualizzata più media/stabile; uno più piccolo rende il training più rumoroso.

### Ricostruisci modello
Crea una nuova rete usando architettura, activation e task correnti. I pesi vengono reinizializzati: il training precedente viene perso, salvo export.

### Avvia
Avvia il loop di training continuo.

### Pausa
Ferma il loop senza cancellare i pesi. Riprendendo, il modello continua dallo stesso stato.

### Epoca
Esegue un solo batch di training e poi si ferma. È utile per osservare lentamente la loss e i parametri.

### Fase
Avanza didatticamente in micro-fasi:

1. Forward
2. Loss
3. Backpropagation
4. Update

È il pulsante più utile per capire la causalità tra predizione, errore, gradiente e modifica dei pesi.

### Reset
Ricrea la rete e azzera la cronologia della loss.

### Canvas Rete neurale live
- linea verde: peso positivo;
- linea rossa: peso negativo;
- spessore: valore assoluto del peso;
- cerchio: neurone;
- valore nel neurone: activation corrente, quando disponibile.

### Dataset / Decision Surface
In regressione mostra punti e curva predetta. In classificazione mostra i punti e uno sfondo che rappresenta la probabilità appresa.

### Loss chart
Mostra l'errore nel tempo. Una curva discendente è un buon segnale, ma va confrontata con validation e test.

### Terminale educativo
Registra eventi di training: forward, loss, backprop, update, cambio modello e seed.

---

## 4. Neuron Lab

Serve a capire un singolo neurone senza il resto della rete.

### Slider x1 e x2
Sono i due input numerici.

### Slider w1 e w2
Sono i pesi che moltiplicano gli input.

### Slider bias
Termine additivo indipendente dagli input.

### Activation selector
Applica tanh, ReLU, sigmoid o linear.

### Formula visualizzata
Il neurone calcola:

`z = x1*w1 + x2*w2 + bias`

poi:

`a = activation(z)`

Questo valore `a` è ciò che viene trasmesso ai neuroni del layer successivo.

---

## 5. Network Builder

Mostra la struttura della rete corrente. La modifica dell'architettura avviene dal campo `Hidden layer sizes` della Dashboard.

Ogni blocco indica il numero di neuroni del layer. Le frecce indicano connessioni Dense: ogni neurone di un layer è collegato a ogni neurone del successivo.

Con `1→4→4→1`:
- Layer 1: 1*4 pesi + 4 bias = 8 parametri.
- Layer 2: 4*4 + 4 = 20 parametri.
- Layer 3: 4*1 + 1 = 5 parametri.
- Totale = 33 parametri.

---

## 6. Teacher vs Student

### Agente A
È il professore. Conosce una regola target, nella demo `y=2.5x+1`, e genera una domanda numerica.

### Agente B
È la rete neurale. Non riceve la formula; vede l'input e produce una predizione.

### Feedback
Dopo la risposta vengono mostrati target e loss, poi viene effettuata backpropagation e update.

### Nuova domanda
Genera un singolo esempio e un singolo update.

### 10 domande automatiche
Ripete dieci volte il meccanismo. Permette di vedere rapidamente il miglioramento.

Conseguenza didattica: mostra che il modello non "capisce" la formula come una regola simbolica, ma modifica parametri numerici per ridurre l'errore sugli esempi.

---

## 7. Train My Data

### Area testo
Formato base: una coppia `x,y` per riga.

Esempio:

```text
1,3
2,5
3,7
4,9
```

### Carica dataset
Trasforma il testo in esempi di regressione e ricostruisce il modello.

### 100 step training
Esegue rapidamente 100 update, utile per vedere se la rete riesce a scoprire una relazione nei tuoi dati.

Attenzione: pochi esempi possono essere memorizzati senza generalizzare. Usa Generalization Lab per controllarlo.

---

## 8. Forward / Backprop Explorer

Questa pagina riassume le quattro fasi.

### Forward
Per ogni Dense layer:

`z = W*x + b`

`a = activation(z)`

### Loss
Misura la differenza tra output e target.

### Backpropagation
Usa la regola della catena per calcolare come cambia la loss rispetto a ogni parametro.

### Update
L'optimizer usa i gradienti per modificare pesi e bias.

---

## 9. Parameter Inspector avanzato

Colonne:

- **Parametro**: nome del peso/bias, per esempio `L2.W[1,3]`.
- **Prima**: valore osservato al controllo precedente.
- **Gradiente**: derivata della loss rispetto a quel parametro.
- **Δ Update**: differenza `dopo-prima`.
- **Dopo**: valore corrente.
- **Tipo**: weight o bias.

Interpretazione:
- gradiente positivo + SGD classico -> il parametro tende a diminuire;
- gradiente negativo -> tende ad aumentare;
- delta quasi zero -> update molto piccolo, gradiente piccolo o Adam che riduce il passo;
- valori che cambiano violentemente -> possibile LR troppo alto.

---

## 10. Dataset & Decision Surface

La Dashboard contiene la visualizzazione completa. In classificazione lo sfondo rappresenta la probabilità prevista su molti punti dello spazio 2D.

Se la frontiera cambia durante il training, stai vedendo direttamente l'effetto collettivo dei pesi sulla funzione appresa.

---

## 11. Computational Graph

Mostra la catena di trasformazioni dell'ultimo forward:

`Input -> z -> activation -> ... -> output -> loss`

- `z`: weighted sum prima dell'activation.
- `a`: output dopo activation.

Questo aiuta a capire che una rete neurale è una composizione di operazioni matematiche e che la backpropagation percorre concettualmente questa composizione in senso inverso.

---

## 12. Model X-Ray

Per ogni layer mostra:
- dimensione input/output;
- activation;
- numero di weights;
- numero di bias;
- parametri totali del layer.

È la sezione da usare per rispondere a domande come: "Perché aggiungere un neurone aumenta così tanto i parametri?".

---

## 13. Generalization Lab

È una delle estensioni più importanti della V2.

### Train %
Percentuale di dati usata realmente per modificare i pesi.

### Validation %
Dati mai usati per l'update. Servono a misurare il comportamento su esempi non direttamente ottimizzati.

### Test % automatico
È `100 - train - validation`. Il test è la verifica finale separata.

### Rigenera split
Mescola nuovamente il dataset e crea nuovi gruppi train/validation/test.

### Train Loss
Errore sul training set.

### Val Loss
Errore su validation.

### Test Loss
Errore sul test.

### Generalization Gap
`Val Loss - Train Loss`.

Un gap crescente mentre la Train Loss continua a scendere può indicare overfitting.

### Accuracy / R²
- Classificazione: accuracy sul test.
- Regressione: R² sul test. Valori più vicini a 1 indicano maggiore capacità di spiegare la variabilità del target.

### Diagnosi
Indicazione didattica: stabile, possibile overfitting, underfitting/training iniziale.

### Curve di generalizzazione
Tre curve permettono di confrontare Train, Validation e Test nel tempo.

### Confusion Matrix
Solo classificazione:
- True Positive;
- False Positive;
- False Negative;
- True Negative.

Mostra errori che l'accuracy da sola nasconde.

---

## 14. Training Diagnostics

### Gradient L2
Norma complessiva dei gradienti. Misura la grandezza globale del segnale di correzione.

### Max |Gradient|
Massimo valore assoluto tra i gradienti. Valori estremi possono indicare exploding gradients.

### Weight L2
Norma dei pesi. Se cresce molto può essere un sintomo di instabilità.

### Dead ReLU
Conta attivazioni zero nei layer ReLU. Un'alta percentuale persistente può significare che molti neuroni non stanno contribuendo.

### Health Monitor
Produce avvisi didattici per:
- exploding gradient;
- vanishing gradient;
- dead ReLU;
- pesi troppo grandi.

### Gradient Flow per layer
Mostra il valore medio assoluto del gradiente per layer. Serve a capire se il segnale di apprendimento si indebolisce andando verso i layer iniziali.

---

## 15. Optimizer Arena

### Epoche confronto
Numero di passaggi per il benchmark.

### LR SGD
Learning rate usato solo dalla copia SGD.

### LR Adam
Learning rate usato solo dalla copia Adam.

### Esegui confronto
Crea due copie con gli stessi identici pesi iniziali e usa lo stesso ordine degli esempi.

Il grafico mostra due curve di loss. Lo scopo non è dichiarare un vincitore universale, ma capire che optimizer e learning rate modificano la dinamica di convergenza.

---

## 16. Mini Transformer / Self-Attention

### Area testo
Contiene la frase da trasformare in token.

### Calcola Attention
Genera embedding deterministici didattici, costruisce Q, K e V semplificati, calcola:

`Attention(Q,K,V) = softmax(Q*K^T / sqrt(d)) * V`

### Token View
Mostra i token e il loro indice.

### Attention Heatmap
Ogni riga indica quanto un token distribuisce la propria attenzione sugli altri token.

Importante: questa sezione spiega il meccanismo centrale della self-attention, ma non implementa un Transformer/GPT completo con training linguistico, multi-head attention reale addestrabile, LayerNorm, residual, tokenizer BPE e language-model objective.

---

## 17. Documentation integrata

Riassume le definizioni principali direttamente nell'app e propone esperimenti.

---

## 18. Python Live

Pagina separata per evitare che lo scroll automatico del codice rubi il focus alla Dashboard.

### Stato Live
Riceve dalla Dashboard:
- epoca;
- fase;
- architettura;
- learning rate;
- optimizer;
- predizione;
- target;
- loss;
- parametri e modello serializzato.

### Esegui Python reale
Carica Pyodide tramite CDN e rifà un forward pass Python usando gli stessi pesi, bias, activation e input della Dashboard. Serve come verifica indipendente della matematica.

La disponibilità di Pyodide richiede accesso Internet alla CDN.

---

## 19. Export / Import e riproducibilità

### Export Model
Il JSON include architettura, task, activation e parametri. Non contiene tutta la cronologia di training.

### Import Model
Ripristina la funzione appresa dal modello. Se il dataset corrente è incompatibile con il numero di input, bisogna caricare un dataset coerente.

### Seed
Permette di ripetere inizializzazione e dataset pseudocasuale, facilitando confronti corretti.

---

## 20. Percorso di studio consigliato

1. Neuron Lab: capire `x*w+b`.
2. Dashboard con rete `1→2→1`, step Fase.
3. Parameter Inspector: osservare un singolo peso.
4. Teacher vs Student: capire supervised learning.
5. Train My Data: insegnare una regola propria.
6. Generalization: capire perché train loss non basta.
7. Diagnostics: capire gradienti e stabilità.
8. Optimizer Arena: confrontare update diversi.
9. XOR / Circles: vedere la non linearità.
10. Mini Transformer: passare dai neuroni alla self-attention.
11. Python Live: verificare lo stesso forward in Python.

---

## 21. Esperimenti pratici

### Esperimento A - Learning Rate
Usa Linear, seed 42, architettura `1→4→1`.
Confronta LR 0.001, 0.03 e 0.1. Osserva tempo di convergenza, oscillazioni e gradienti.

### Esperimento B - Capacità
Su Sine prova `1→2→1`, poi `1→8→8→1`. Confronta Train e Validation Loss.

### Esperimento C - XOR
Con `2→1` la rete non dispone di hidden layer sufficiente per una frontiera non lineare. Usa `2→4→1` e osserva la decision surface.

### Esperimento D - Dead ReLU
Seleziona ReLU, usa LR alto e osserva `Dead ReLU`. Poi resetta con LR più basso.

### Esperimento E - Overfitting
Usa pochi dati personalizzati e una rete molto grande. Continua il training e osserva il Generalization Gap.

### Esperimento F - SGD vs Adam
Mantieni seed e architettura fissi. Esegui Optimizer Arena con diversi LR. Nota che il confronto corretto richiede stessi pesi e stessi dati.

---

## 22. Glossario

- **Input**: dato fornito al modello.
- **Target**: risposta corretta durante supervised learning.
- **Prediction**: output prodotto dal modello.
- **Weight/Peso**: parametro moltiplicativo di una connessione.
- **Bias**: parametro additivo di un neurone.
- **Parametro**: qualsiasi valore trainabile, tipicamente peso o bias.
- **Activation**: funzione non lineare applicata a z.
- **Forward Pass**: calcolo input -> output.
- **Loss**: misura dell'errore.
- **Gradient**: derivata della loss rispetto a un parametro.
- **Backpropagation**: calcolo efficiente dei gradienti attraverso i layer.
- **Optimizer**: algoritmo che aggiorna i parametri.
- **Learning Rate**: dimensione base del passo di update.
- **Batch**: gruppo di esempi elaborato prima del prossimo update/step visualizzato.
- **Epoca**: nel progetto, ciclo visualizzato di training; in letteratura spesso indica un passaggio su tutto il training set.
- **Overfitting**: ottime prestazioni sul train ma peggiori su dati nuovi.
- **Underfitting**: modello incapace di rappresentare bene la relazione.
- **Generalizzazione**: capacità di funzionare su esempi non visti durante l'update.
- **Decision Surface**: mappa delle predizioni di classificazione nello spazio degli input.
- **Embedding**: vettore numerico associato a un token/oggetto.
- **Attention**: meccanismo che combina rappresentazioni pesandole in base a relazioni Q/K.

---

## 23. Due modalità: Didattica e PyTorch Advanced

A partire dalla V3, AI Model Lab non impone più un unico livello di complessità. Il selettore globale **Didattica / PyTorch Pro** permette di scegliere tra due motori differenti.

### Modalità Didattica

È la modalità visuale originale, progettata per rendere ogni passaggio leggibile:

- motore MLP JavaScript piccolo e sincrono;
- Forward, Loss, Backpropagation e aggiornamento pesi osservabili passo per passo;
- dataset didattici;
- computational graph e Parameter Inspector leggibili;
- Mini Transformer semplificato per studiare Q/K/V e attention.

Questa modalità resta intenzionalmente semplice perché serve a capire **perché** un parametro cambia.

### Modalità PyTorch Advanced

È un secondo ambiente realmente eseguito in Python attraverso un backend FastAPI + PyTorch. Include:

- `torch.nn.Module` reali;
- autograd reale (`loss.backward()`);
- optimizer PyTorch (`SGD`, `Adam`, `AdamW`);
- weight decay e gradient clipping;
- CPU o CUDA quando disponibile;
- Automatic Mixed Precision su CUDA;
- forward hooks per analizzare le activation;
- Tensor / Parameter Inspector;
- salvataggio e caricamento di checkpoint `.pt`;
- Transformer language model autoregressivo reale;
- byte-level tokenizer UTF-8;
- token embedding e positional embedding;
- causal Multi-Head Self-Attention;
- residual connection, LayerNorm e GELU MLP;
- Cross-Entropy Loss;
- perplexity, token/s, gradient norm e attention map;
- generazione autoregressiva con temperature e top-k.

Il Transformer della modalità Advanced è un **vero language model addestrabile**, non una heatmap simulata. Tuttavia, la parola *Large* in LLM indica anche una scala molto grande. Un modello locale con poche centinaia di migliaia o milioni di parametri usa gli stessi meccanismi fondamentali, ma non ha automaticamente le capacità di un modello frontier addestrato su miliardi di token e grandi cluster GPU.

### Perché mantenere entrambe

La modalità PyTorch mostra *come si costruisce un sistema reale*; la modalità Didattica mostra *cosa sta accadendo dentro quel sistema*. Passare continuamente da una modalità all'altra è parte del metodo didattico del laboratorio.
## 24. Modalità Framework Pro V3 compatibile

La V4 conserva integralmente le pagine PyTorch Engine, Transformer LM classico e Torch X-Ray della V3. Servono per fare il passaggio intermedio tra il motore didattico e il Modern Transformer V4.

### PyTorch Engine
Crea un `torch.nn.Module` MLP reale con `nn.Linear`, activation, Dropout, SGD/Adam/AdamW, weight decay, gradient clipping, CPU/CUDA e AMP. I forward hook raccolgono media, deviazione standard, min/max e zero fraction delle activation.

### Transformer LM classico
Implementa un decoder autoregressivo byte-level con token/position embedding, LayerNorm, Q/K/V, causal multi-head attention, residual, GELU MLP, LM head, Cross Entropy e `loss.backward()`. È utile come architettura di riferimento prima del blocco moderno V4.

### Torch X-Ray
Mostra `nn.Module` tree, shape/statistiche dei tensor, gradienti, attention snapshot e checkpoint `.pt`.

---

## 25. Ultimate LLM Studio V4 - ogni gruppo di controlli

### Architettura
- **d_model**: larghezza del vettore hidden. Aumentandolo crescono quasi quadraticamente molti pesi delle proiezioni.
- **Layers**: numero di Transformer block. Aumenta profondità, parametri, attivazioni e tempo di training.
- **Attention heads**: numero di query head. `d_model` deve essere divisibile per questo valore.
- **KV heads (GQA)**: numero di key/value head. Deve dividere il numero di attention head. Valori inferiori implementano Grouped Query Attention e riducono parametri/KV memory rispetto a MHA piena.
- **Context**: massimo numero di token visti contemporaneamente. La self-attention standard ha costo quadratico rispetto alla lunghezza di contesto.
- **MLP ratio**: dimensione hidden della SwiGLU rispetto a `d_model`.
- **Dropout**: regolarizzazione stocastica durante training; in eval/generation è disattivata.
- **RoPE theta**: base delle frequenze delle Rotary Positional Embeddings. Cambiarla modifica la geometria posizionale applicata a Q e K.

### Tokenizer
- **Byte UTF-8**: vocabolario fisso da 256 byte; nessun training tokenizer, robusto a qualunque testo UTF-8 ma sequenze più lunghe.
- **Trainable Byte-BPE**: parte dai 256 byte e apprende merge frequenti sul corpus. `BPE vocab` stabilisce la dimensione obiettivo massima.

### LoRA
- **LoRA rank**: se maggiore di zero aggiunge matrici low-rank A/B alle proiezioni attention/MLP.
- **LoRA alpha**: scala l’adattatore con `alpha/rank`.
- **Freeze base con LoRA**: rende non trainabili i pesi base dei layer LoRA e aggiorna principalmente gli adapter. Il contatore `Trainable` mostra la riduzione effettiva.

### Efficienza
- **Gradient checkpointing**: ricalcola parti del forward durante backward per ridurre memoria delle activation, aumentando il calcolo.
- **PyTorch SDPA**: usa `scaled_dot_product_attention`; PyTorch sceglie il kernel disponibile. Su hardware compatibile può usare implementazioni ottimizzate.
- **torch.compile**: richiede supporto della build PyTorch; può ridurre overhead dopo una fase di compilazione, ma non è sempre vantaggioso per modelli minuscoli.
- **AMP dtype**: `auto`, BF16, FP16 o FP32. FP16 usa GradScaler su CUDA; BF16 dipende dal supporto hardware.

### Ottimizzazione
- **LR**: learning rate massimo.
- **Min LR**: limite inferiore del cosine decay.
- **Warmup steps**: il learning rate cresce linearmente da quasi zero a LR.
- **LR decay steps**: orizzonte del cosine schedule.
- **Weight decay**: penalizzazione dei pesi applicata dall’optimizer.
- **Grad clip**: limite della norma globale del gradiente.
- **Grad accumulation**: somma gradienti di più micro-batch prima di `optimizer.step()`, simulando un batch effettivo maggiore.
- **Seed**: riproducibilità pseudocasuale.
- **Device**: Auto, CPU o CUDA.
- **Validation %**: porzione di token non usata per gli update, riservata a stime di validation loss/perplexity.

### Training Job non bloccante
**Steps** indica quanti optimizer step richiedere. **Batch** è il numero di sequenze per micro-batch. **Eval ogni** decide la frequenza della validation. **Avvia Job** crea un thread backend; **Stop Job** imposta un evento di arresto. La progress bar è soltanto monitoraggio: il training vero rimane nel backend Python.

### Metriche
- **Parametri**: numero totale di elementi trainabili + congelati.
- **Trainable**: sottoinsieme con `requires_grad=True`; con LoRA+freeze può essere molto più piccolo del totale.
- **Train Loss**: cross-entropy dell’ultimo step.
- **Val Loss**: cross-entropy media su batch di validation.
- **Tokens/s**: throughput osservato del training step; dipende da hardware, contesto, batch e precisione.
- **LR**: valore effettivo dello scheduler al passo corrente.
- **Gradient L2**: norma L2 dei gradienti prima del clipping.

### Generazione
- **Temperature**: divide i logits prima di softmax; valori bassi rendono la distribuzione più concentrata.
- **Top-k**: conserva al massimo i k token con logit maggiore. `0` disattiva il filtro.
- **Top-p**: nucleus sampling; conserva il più piccolo insieme ordinato di token la cui probabilità cumulativa raggiunge p.
- **Repetition penalty**: modifica i logits dei token già presenti nel contesto per ridurre o aumentare la ripetizione.

### Tokenizer & Embedding Microscope
`Tokenizza` mostra ID, testo e byte di ciascun token. `Embedding` estrae i vettori reali dalla matrice `token_embedding.weight`, li normalizza e visualizza la similarità coseno token-token.

### Checkpoint e Experiment
`Checkpoint .pt` salva model state, optimizer state, tokenizer, step, history e token split. `Salva esperimento` registra config e metriche in SQLite, utile per confrontare run senza doverli ricordare a memoria.

---

## 26. Autograd Microscope

Questa pagina implementa reverse-mode autodiff scalare direttamente nel browser. Il grafo predefinito è:

`x,w → x*w → +b → tanh → -target → errore → quadrato → *0.5 → loss`

Ogni nodo memorizza `data`, `grad`, dipendenze, operazione e regola `_backward` locale.

**Costruisci Forward Graph** calcola i valori. **Backward()** ordina topologicamente il DAG e applica le regole in ordine inverso. **Gradient Descent Update** usa `w <- w - lr*dw` e `b <- b - lr*db`.

Questa pagina non sostituisce `torch.autograd`: rende visibile il principio che PyTorch generalizza a tensor e grafi molto più grandi.

---

## 27. Scale & Memory Lab

Questa pagina NON alloca il modello selezionato: usa formule analitiche per stimare parametri, memoria e FLOPs. È quindi sicuro esplorare configurazioni da miliardi di parametri anche su un PC normale.

Le stime includono embedding, attention GQA, SwiGLU, norm, pesi, gradienti, stato optimizer e una stima delle activation. Le celle `rough` sono approssimazioni: kernel SDPA/Flash, allocator, KV cache, master weights e checkpointing possono cambiare il consumo reale.

---

## 28. Experiment Tracker

Gli esperimenti vengono salvati localmente in `backend/experiments_v4.sqlite3`. Ogni record contiene nome, data, configurazione, step, metriche, numero di parametri e note.

---

## 29. Cosa significa “framework reale” e cosa non significa

Framework Pro usa realmente PyTorch, `nn.Module`, tensor, autograd, optimizer, CUDA se disponibile, AMP, checkpoint e training autoregressivo. Modern Transformer V4 usa blocchi architetturali presenti nei decoder moderni.

Non significa automaticamente cluster multi-node, dataset web-scale, migliaia di GPU, distributed sharding, fault tolerance datacenter, serving ad alta concorrenza o valutazioni di sicurezza necessarie a un prodotto commerciale.

---

## 30. Inventario esaustivo dei controlli UI

Questa tabella è generata direttamente da `index.html` V4 e copre ogni pulsante, input, select e textarea con ID.

| Pagina | Tipo | ID | Label/Testo | Default | Effetto |
|---|---|---|---|---|---|
| Globale / barra laterale / header | button | `modeDidacticBtn` | Didattica | azione | Seleziona il motore didattico JavaScript, privilegiando trasparenza e micro-step. |
| Globale / barra laterale / header | button | `modeAdvancedBtn` | Framework Pro V4 | azione | Seleziona l’area Framework Pro: backend FastAPI/PyTorch reale e pagine avanzate V3/V4. |
| Globale / barra laterale / header | button | `openPythonBtn` | Python Live | azione | Apre Python Live in una finestra separata, sincronizzata con lo stato del laboratorio. |
| Globale / barra laterale / header | button | `exportModelBtn` | Export Model | azione | Esporta il modello didattico in JSON. |
| Globale / barra laterale / header | button | `importModelBtn` | Import Model | azione | Apre il selettore file per importare un modello JSON compatibile. |
| Globale / barra laterale / header | input/file | `importModelFile` |  |  | Selettore file locale usato dal comando di importazione. |
| Globale / barra laterale / header | button | `exportDataBtn` | Export Dataset | azione | Esporta il dataset didattico corrente in JSON. |
| Globale / barra laterale / header | button | `tourBtn` | Tour guidato | azione | Avvia il tour guidato dell’interfaccia didattica. |
| Globale / barra laterale / header | button | `helpModeBtn` | Help contestuale | azione | Attiva/disattiva l’help contestuale e richiama tooltip/legenda. |
| Training Controls | select | `taskMode` | Task | regression | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Training Controls | select | `datasetSelect` | Dataset | regressionLinear | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Training Controls | input/text | `hiddenSizes` | Hidden layer sizes | 4,4 | Campo testuale di configurazione o input. |
| Training Controls | select | `activationSelect` | Activation | tanh | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Training Controls | select | `optimizerSelect` | Optimizer | adam | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Training Controls | input/number | `seedInput` |  | 42 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Training Controls | button | `applySeedBtn` | Applica | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | input/range | `lrSlider` | Learning rate 0.030 | 0.03 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Training Controls | input/range | `speedSlider` | Velocità 250 ms | 250 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Training Controls | input/number | `batchSize` | Batch size | 8 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Training Controls | button | `rebuildBtn` | Ricostruisci modello | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | button | `startBtn` | Avvia | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | button | `pauseBtn` | Pausa | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | button | `stepEpochBtn` | Epoca | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | button | `stepPhaseBtn` | Fase | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | button | `resetBtn` | Reset | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Training Controls | button | `clearLog` | Pulisci | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Dentro un singolo neurone | input/range | `nx1` | x1 | 1.2 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Dentro un singolo neurone | input/range | `nx2` | x2 | -.7 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Dentro un singolo neurone | input/range | `nw1` | w1 | .8 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Dentro un singolo neurone | input/range | `nw2` | w2 | -1.1 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Dentro un singolo neurone | input/range | `nb` | bias | .2 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Dentro un singolo neurone | select | `nact` |  | tanh | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Agente A ↔ Agente B | button | `teacherStepBtn` | Nuova domanda | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Agente A ↔ Agente B | button | `teacherAutoBtn` | 10 domande automatiche | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Train My Data | textarea | `customData` |  | 1,3 2,5 3,7 4,9 5,11 6,13 | Area dati/testo usata dalla funzione o dal training della pagina corrente. |
| Train My Data | button | `loadCustomBtn` | Carica dataset | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Train My Data | button | `trainCustomBtn` | 100 step training | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Generalization Lab | button | `resplitBtn` | Rigenera split | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Generalization Lab | input/range | `trainPct` | Train % 70% | 70 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Generalization Lab | input/range | `valPct` | Validation % 15% | 15 | Slider numerico: modifica il valore indicato in tempo reale o per il prossimo step. |
| Optimizer Arena | button | `runOptimizerArena` | Esegui confronto | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Optimizer Arena | input/number | `arenaEpochs` | Epoche confronto | 80 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Optimizer Arena | input/number | `arenaSgdLr` | LR SGD | 0.03 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Optimizer Arena | input/number | `arenaAdamLr` | LR Adam | 0.01 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | button | `advRefreshHealth` | Verifica backend | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| PyTorch Advanced Engine | select | `torchDevice` | Device richiesto | auto | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| PyTorch Advanced Engine | input/checkbox | `torchAmp` | Automatic Mixed Precision Usa AMP FP16 su CUDA; su CPU resta disattivata automaticamente. | ON | Flag booleano: abilita o disabilita la funzione indicata. |
| PyTorch Advanced Engine | select | `torchMlpDataset` | Dataset | linear | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| PyTorch Advanced Engine | input/number | `torchMlpInput` | Input | 1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/number | `torchMlpOutput` | Output | 1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/text | `torchMlpHidden` | Hidden sizes | 32,32 | Campo testuale di configurazione o input. |
| PyTorch Advanced Engine | select | `torchMlpTask` | Task | regression | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| PyTorch Advanced Engine | select | `torchMlpActivation` | Activation | relu | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| PyTorch Advanced Engine | input/number | `torchMlpDropout` | Dropout | 0.0 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | select | `torchMlpOptimizer` | Optimizer | adamw | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| PyTorch Advanced Engine | input/number | `torchMlpBatch` | Batch | 32 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/number | `torchMlpLr` | Learning rate | 0.001 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/number | `torchMlpWd` | Weight decay | 0.0001 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/number | `torchMlpClip` | Gradient clip | 1.0 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/number | `torchMlpSeed` | Seed | 42 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | input/number | `torchMlpSpeed` | Intervallo realtime (ms) | 180 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| PyTorch Advanced Engine | button | `torchCreateMlp` | Crea nn.Module | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| PyTorch Advanced Engine | button | `torchMlpStep` | 1 step | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| PyTorch Advanced Engine | button | `torchMlp50` | 50 step | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| PyTorch Advanced Engine | button | `torchMlpStart` | Realtime | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| PyTorch Advanced Engine | button | `torchMlpPause` | Pausa | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Transformer Language Model reale | input/number | `llmDModel` | d_model | 128 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmHeads` | Attention heads | 4 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmLayers` | Transformer blocks | 4 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmContext` | Context length | 64 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmRatio` | MLP ratio | 4 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmDropout` | Dropout | 0.1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | select | `llmOptimizer` | Optimizer | adamw | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Transformer Language Model reale | input/number | `llmBatch` | Batch | 8 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmLr` | Learning rate | 0.0003 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmWd` | Weight decay | 0.01 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmClip` | Gradient clip | 1.0 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmSeed` | Seed | 42 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/checkbox | `llmTie` | Tie token embedding ↔ lm_head | ON | Flag booleano: abilita o disabilita la funzione indicata. |
| Transformer Language Model reale | input/number | `llmSpeed` | Realtime interval (ms) | 80 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | button | `torchCreateLlm` | Tie token embedding ↔ lm_head | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Transformer Language Model reale | button | `llmTrainStepBtn` | 1 step | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Transformer Language Model reale | button | `llm20` | 20 step | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Transformer Language Model reale | button | `llmStart` | Realtime | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Transformer Language Model reale | button | `llmPause` | Pausa | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Transformer Language Model reale | textarea | `llmTrainingText` |  | L'intelligenza artificiale apprende regolarità statistiche dai dati. Un language… | Area dati/testo usata dalla funzione o dal training della pagina corrente. |
| Transformer Language Model reale | input/text | `llmPrompt` |  | L'intelligenza | Campo testuale di configurazione o input. |
| Transformer Language Model reale | input/number | `llmGenTokens` | New tokens | 80 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmTemperature` | Temperature | 0.8 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | input/number | `llmTopK` | Top-k | 40 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Transformer Language Model reale | button | `llmGenerate` | Genera | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Torch X-Ray | button | `torchRefreshXray` | Aggiorna snapshot | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Torch X-Ray | input/text | `checkpointName` |  | experiment | Campo testuale di configurazione o input. |
| Torch X-Ray | button | `saveCheckpoint` | Salva | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Torch X-Ray | button | `refreshCheckpoints` | Aggiorna | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Ultimate LLM Studio V4 | button | `v4HealthBtn` | Aggiorna sistema | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |
| Ultimate LLM Studio V4 | input/number | `v4DModel` | d_model | 192 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Layers` | Layers | 6 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Heads` | Attention heads | 6 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4KvHeads` | KV heads (GQA) | 2 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Context` | Context | 256 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Ratio` | MLP ratio | 4 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Dropout` | Dropout | 0.05 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Rope` | RoPE θ | 10000 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | select | `v4Tokenizer` | Tokenizer | byte | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Ultimate LLM Studio V4 | input/number | `v4BpeVocab` | BPE vocab | 384 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4LoraRank` | LoRA rank | 0 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4LoraAlpha` | LoRA alpha | 16 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/checkbox | `v4FreezeBase` | Freeze base con LoRA | OFF | Flag booleano: abilita o disabilita la funzione indicata. |
| Ultimate LLM Studio V4 | input/checkbox | `v4GradCkpt` | Gradient checkpointing | OFF | Flag booleano: abilita o disabilita la funzione indicata. |
| Ultimate LLM Studio V4 | input/checkbox | `v4Sdpa` | PyTorch SDPA | ON | Flag booleano: abilita o disabilita la funzione indicata. |
| Ultimate LLM Studio V4 | input/checkbox | `v4Compile` | torch.compile | OFF | Flag booleano: abilita o disabilita la funzione indicata. |
| Ultimate LLM Studio V4 | select | `v4Optimizer` | Optimizer | adamw | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Ultimate LLM Studio V4 | select | `v4Amp` | AMP dtype | auto | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Ultimate LLM Studio V4 | input/number | `v4Lr` | LR | 0.0003 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4MinLr` | Min LR | 0.00003 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Warmup` | Warmup steps | 20 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Decay` | LR decay steps | 2000 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Wd` | Weight decay | 0.1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Clip` | Grad clip | 1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Accum` | Grad accumulation | 1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Seed` | Seed | 42 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | select | `v4Device` | Device | auto | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Ultimate LLM Studio V4 | input/number | `v4ValPct` | Validation % | 10 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | button | `v4CreateBtn` | Crea Modern Transformer | azione | Crea realmente sul backend un ModernTransformerLM PyTorch con la configurazione V4 corrente. |
| Ultimate LLM Studio V4 | textarea | `v4Corpus` |  | AI Model Lab V4 è un laboratorio per capire e sperimentare reti neurali e langua… | Area dati/testo usata dalla funzione o dal training della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4JobSteps` | Steps | 200 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Batch` | Batch | 8 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Eval` | Eval ogni | 20 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | button | `v4StepBtn` | 1 step | azione | Esegue un singolo optimizer step V4, includendo i micro-batch di gradient accumulation configurati. |
| Ultimate LLM Studio V4 | button | `v4StartJobBtn` | Avvia Job | azione | Avvia un training job in background sul backend; la UI continua a rimanere responsiva. |
| Ultimate LLM Studio V4 | button | `v4StopJobBtn` | Stop Job | azione | Imposta lo stop event del job V4; il thread termina al successivo punto di controllo. |
| Ultimate LLM Studio V4 | input/text | `v4TokenText` |  | Transformer attention modello | Campo testuale di configurazione o input. |
| Ultimate LLM Studio V4 | button | `v4TokenBtn` | Tokenizza | azione | Mostra la tokenizzazione reale del tokenizer attivo (byte o BPE addestrato). |
| Ultimate LLM Studio V4 | button | `v4EmbedBtn` | Embedding | azione | Calcola la matrice di similarità coseno tra gli embedding dei token del testo inserito. |
| Ultimate LLM Studio V4 | textarea | `v4Prompt` |  | AI Model Lab | Area dati/testo usata dalla funzione o dal training della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4GenTokens` | New tokens | 120 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Temp` | Temperature | 0.8 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4TopK` | Top-k | 50 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4TopP` | Top-p | 0.95 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | input/number | `v4Rep` | Repetition penalty | 1.05 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Ultimate LLM Studio V4 | button | `v4GenerateBtn` | Genera autoregressivamente | azione | Esegue generazione autoregressiva con temperature, top-k, top-p e repetition penalty. |
| Ultimate LLM Studio V4 | input/text | `v4SaveName` |  | ultimate_experiment | Campo testuale di configurazione o input. |
| Ultimate LLM Studio V4 | textarea | `v4Notes` |  |  | Area dati/testo usata dalla funzione o dal training della pagina corrente. |
| Ultimate LLM Studio V4 | button | `v4CheckpointBtn` | Checkpoint .pt | azione | Salva un checkpoint .pt V4 con modello, optimizer, tokenizer, step e dataset tokenizzato. |
| Ultimate LLM Studio V4 | button | `v4ExperimentBtn` | Salva esperimento | azione | Registra configurazione e metriche correnti nel database SQLite degli esperimenti. |
| Autograd Microscope | input/number | `agX` | x | 2 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Autograd Microscope | input/number | `agW` | w | -1.5 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Autograd Microscope | input/number | `agB` | b | 0.5 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Autograd Microscope | input/number | `agTarget` | target | 1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Autograd Microscope | button | `agForwardBtn` | Costruisci Forward Graph | azione | Costruisce un computational DAG scalare per x·w+b → tanh → errore²/2. |
| Autograd Microscope | button | `agBackwardBtn` | Backward() | azione | Esegue reverse-mode autodiff sul DAG e mostra il gradiente globale di ogni nodo. |
| Autograd Microscope | button | `agUpdateBtn` | Gradient Descent Update | azione | Applica un update di gradient descent a w e b usando i gradienti calcolati. |
| Autograd Microscope | input/number | `agLr` | Learning rate | 0.1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | select | `scalePreset` | Preset | custom | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Scale & Memory Lab | input/number | `scaleVocab` | Vocab | 32000 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleD` | d_model | 1024 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleHeads` | Heads | 16 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleKv` | KV heads | 4 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleLayers` | Layers | 24 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleContext` | Context | 4096 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleRatio` | MLP ratio | 4 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | input/number | `scaleBatch` | Batch | 1 | Parametro numerico: influenza configurazione, training, visualizzazione o stima della pagina corrente. |
| Scale & Memory Lab | select | `scalePrecision` | Precision bytes | 2 | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Scale & Memory Lab | select | `scaleOptimizer` | Optimizer | adamw | Seleziona una modalità/opzione; l’effetto viene applicato quando il relativo modello viene ricostruito o il comando associato viene eseguito. |
| Scale & Memory Lab | button | `scaleCalcBtn` | Calcola scala | azione | Calcola parametri, memoria e FLOPs approssimativi senza allocare realmente il modello. |
| Experiment Tracker | button | `expRefreshBtn` | Aggiorna | azione | Rilegge da SQLite la lista degli esperimenti salvati. |
| Mini Transformer / Self-Attention | textarea | `attentionText` |  | il modello impara quali parole sono importanti nel contesto | Area dati/testo usata dalla funzione o dal training della pagina corrente. |
| Mini Transformer / Self-Attention | button | `attentionRunBtn` | Calcola Attention | azione | Esegue l’azione descritta dal testo del pulsante nella sezione corrente. |

---

## 31. Percorso di studio consigliato

1. Neuron Lab. 2. Autograd Microscope. 3. Dashboard 1→2→1. 4. Fase Forward/Loss/Backprop/Update. 5. Parameter Inspector. 6. Teacher vs Student. 7. Generalization e Diagnostics. 8. PyTorch Engine. 9. Transformer LM classico. 10. Ultimate LLM V4. 11. Tokenizer/Embedding microscope. 12. Scale & Memory Lab. 13. Torch X-Ray e checkpoint. 14. Experiment Tracker.
