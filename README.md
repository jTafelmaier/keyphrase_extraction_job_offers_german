


#### Python keyphrase extraction from german job offers



### NOTES
- Use pip to install all dependencies in file "requirements.txt".
- "_app_test_keywords.py" is the main script from which most other relevant functions can be called.
- All contents of the "lib" directory were not developed as part of the bachelor thesis and are merely used.


content of "_app_test_keywords.py":

```python

text = "Durch stetiges Wachstum und kontinuierliche Weiterentwicklung sind wir am zentralen Standort Innsbruck auf der Suche nach Verstärkung:\n" \
    "\n" \
    "Junior Programmierer:in in Teilzeit\n" \
    "(Geringfügig bis 25h)\n" \
    "Deine Aufgaben\n" \
    "Webscraping (Analyse von Webseiten)\n" \
    "Entwickeln von Tools unterschiedlicher Art (v.A. via API)\n" \
    "Dein Profil\n" \
    "Eine oder mehrere Programmiersprachen, idealerweise Python\n" \
    "Grundverständnis einer Markup-Language wie HTML\n" \
    "Eigenständige Arbeitsweise und sauberer Programmierstil\n" \
    "Von Vorteil\n" \
    "Kenntnisse von Algorithmen und Datenstrukturen\n" \
    "Kenntnisse in GIT, HTTPS, CSS\n" \
    "Funktionaler Programmierstil\n" \
    "Unser Angebot\n" \
    "Stundenausmaß deiner Wahl und zeitliche Flexibilität\n" \
    "Selbstbestimmtes Arbeiten: du kannst deine eigenen Ideen umsetzen, wir geben dir wenig Vorgaben\n" \
    "Möglichkeiten, dich im Bereich Machine Learning oder Webdesign weiterzuentwickeln\n" \
    "Arbeiten mit den modernen Firmen-Websites unserer Kunden, Einblick in deren Entwicklung\n" \
    "Wir bieten dir einen sicheren Arbeitsplatz mit einem interessanten Arbeitsumfeld, abwechslungsreichen Projekten und Herausforderungen. Am gut angebundenen Standort direkt beim Innsbrucker Hauptbahnhof freut sich ein junges und freundliches Team auf die Zusammenarbeit.\n" \
    "\n" \
    "Du wirst gezielt eingeschult und bei uns stehen die Türen jederzeit offen, falls dir mal etwas unklar ist. Gerne lassen wir dich an unserem Wissen teilhaben und begleiten dich\n" \
    "\n" \
    "Je nach deiner Qualifikation, Erfahrung und der Motivation, an der Wachstumsphase eines aufstrebenden, innovativen Unternehmen mitzuwirken, zeigen wir die Bereitschaft zur Überzahlung auf Basis des Kollektivvertrages Information und Consulting.\n" \
    "\n" \
    "Falls dein Interesse geweckt wurde,\n" \
    "freut sich das gesamte Team sehr auf deine Kontaktaufnahme!"

print("predicted keywords:")

for text_keyword in _model_T5.get_list_lists_texts_keywords_load_model([text])[0]:
    print(" - " + text_keyword)
```

### result of executing "_app_test_keywords.py":

```bash
predicted keywords:
 - Junior Programmierer
 - Junior Programmiererin
 - Programmierer
 - Programmiererin
 - Programmierung
 - Teilzeit
 - Python
 - Webscraping
 - HTML
 - HTTPS
 - CSS
 - Machine Learning
 - Webdesign
 - Vollzeit
 - Teilzeitjob
 - Nebenjob
 - Studentenjob
 - Tiroler Unterland
```

