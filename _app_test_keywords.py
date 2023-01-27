

"""
Script to evaluate predictions of models for keyword generation.
"""


import pandas as pd

from lib.unary import _dicts
from lib.unary import _iters

from src.ml import _ml_descriptive_statistics
from src.ml import _ml_evaluation
from src.ml import _ml_retrieval
# from src.ml.models import _model_bart
# from src.ml.models import _model_bert
from src.ml.models import _model_T5
# from src.ml.models import _model_kea
# from src.ml.models import _model_keybert
# from src.ml.models import _model_rake
# from src.ml.models import _model_tf_idf




def main():

    """
    Function to execute the task described in the file documentation.
    """

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

#     _model_T5.train_new_model()
# 
#     dataframe_data_testing = _ml_retrieval.get_dataframe_data_partition(_ml_retrieval.PATH_CSV_DATA_TESTING)
# 
#     list_lists_texts_keywords_target = _ml_retrieval.get_list_lists_texts_keywords(dataframe_data_testing)
# 
#     list_htmls = _ml_retrieval.get_list_htmls(dataframe_data_testing)
# 
#     list_lists_texts_keywords_predicted = _model_T5.get_list_lists_texts_keywords_load_model(list_htmls)
# 
#     dataframe_evaluation = _ml_evaluation.get_dataframe_evaluation(
#             iterable_htmls=list_htmls,
#             list_lists_texts_keywords_predicted=list_lists_texts_keywords_predicted,
#             list_lists_texts_keywords_target=list_lists_texts_keywords_target)
# 
#     text_metrics_average = _ml_evaluation.get_dict_metrics_average(dataframe_evaluation) \
#         >> _dicts.to_iterable_pairs() \
#         >> _iters.to_iterable_mapped(_iters.to_text_joined("")) \
#         >> _iters.to_text_joined("\n")
# 
#     print(text_metrics_average)
# 
#     _ml_evaluation.plot_correlations_of_f1(
#             dataframe_evaluation=dataframe_evaluation,
#             bool_plot_distributions=True)


if __name__ == "__main__":
    main()

