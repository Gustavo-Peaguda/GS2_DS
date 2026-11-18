# AEGIS — Data Science (Global Solution 2026.2)

Previsão de dengue com 4 semanas de antecedência em São Paulo, Osasco, Santo André, São Bernardo do Campo e São Caetano do Sul, com dados do InfoDengue.

## Arquivos

| Arquivo | O que é |
|---|---|
| `AEGIS_DataScience_GS.ipynb` | notebook executado (entrega principal) |
| `dengue_infodengue_2018_2026_5municipios.csv` | base original do InfoDengue |
| `dengue_base_tratada.csv` | base depois do tratamento (gerada pelo notebook) |
| `fonte_dos_dados.txt` | fonte e links dos dados |
| `integrantes.txt` | nomes e RMs da equipe |
| `app.py` | painel Streamlit (extra) |
| `metricas_notebook.json` | métricas do notebook, usadas pelo painel para conferir os resultados |
| `requirements.txt` | bibliotecas necessárias |

## Como rodar o painel

**Importante:** os comandos precisam começar com `py -m` ou `python -m`. Digitar só `pip` ou só `streamlit` costuma dar o erro "não é reconhecido como nome de cmdlet" no Windows.

No terminal, dentro desta pasta:

1. Instale as bibliotecas (só na primeira vez):

```
py -m pip install -r requirements.txt
```

2. Abra o painel:

```
py -m streamlit run app.py
```

O painel abre no navegador. Para fechar, aperte Ctrl + C no terminal.

**Se o `py` não for reconhecido**, use `python` no lugar:

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Se nenhum dos dois funcionar, instale o Python em python.org marcando a opção **"Add python.exe to PATH"** e abra o terminal de novo.