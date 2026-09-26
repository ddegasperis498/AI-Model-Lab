# Workflow GitHub consigliato

Repository dedicato consigliato: `AI-Model-Lab` (private).

Branch:
- `main`: versione stabile
- `develop`: integrazione
- `feature/...`: nuove funzioni
- `fix/...`: correzioni

Per ogni modifica:
1. creare branch;
2. implementare;
3. eseguire `RUN_V41_REGRESSION_TESTS.bat` + self-test;
4. aprire PR verso `develop`/`main`;
5. mantenere release notes.

Non versionare:
- checkpoint `.pt` pesanti;
- database esperimenti locali;
- `.env`/segreti;
- virtual environment.

Il `.gitignore` incluso copre questi casi principali.
