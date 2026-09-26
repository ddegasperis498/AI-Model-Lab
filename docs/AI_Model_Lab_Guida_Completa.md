# AI Model Lab V2 - Guida completa all'interfaccia e ai concetti

## 1. Obiettivo del progetto

AI Model Lab V2 è un laboratorio didattico interattivo progettato per rendere visibili i meccanismi interni di un modello neurale: input, pesi, bias, attivazioni, forward pass, loss, gradienti, backpropagation, update dei parametri, generalizzazione e diagnostica del training.

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

## 23. Limiti intenzionali del laboratorio

AI Model Lab è progettato per essere comprensibile. Non cerca di sostituire framework come PyTorch/TensorFlow né implementa un LLM industriale. Alcune scelte sono volutamente semplificate:

- motore MLP piccolo e sincrono;
- training nel thread del browser;
- dataset didattici;
- attention semplificata e non addestrata come un Transformer completo;
- diagnostica con soglie euristiche pensate per apprendimento, non per produzione.

Il vantaggio è che ogni passaggio resta leggibile e collegabile alla matematica.
