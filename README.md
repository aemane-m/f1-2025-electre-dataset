# F1 2025 – Dataset et implémentation ELECTRE IS

Ce dépôt contient les données et le script Python utilisés pour une analyse de décision multicritère (MCDA) basée sur la méthode ELECTRE IS, appliquée au calendrier de la saison de Formule 1 2025.

L’objectif de ce projet est de comparer les Grands Prix de la saison 2025 selon des critères logistiques, économiques et sportifs, dans une logique de décision incrémentale (optimisation du choix du Grand Prix suivant).

## Contenu du dépôt

f1_2025_variables_brutes.csv Jeu de données contenant les variables brutes utilisées pour la construction des critères d'évaluation.

electre_is_incremental.py Script Python implémentant la méthodologie ELECTRE IS adaptée à une logique séquentielle/incrémentale.

## Description du dataset

Le fichier f1_2025_variables_brutes.csv regroupe les informations suivantes pour chaque épreuve du calendrier 2025 :

Identification : Nom du Grand Prix et du circuit.

Géographie : Coordonnées (latitude, longitude), pays et macrorégion (Europe, Amérique, Asie, Moyen-Orient, Océanie).

Temporel : Date officielle dans le calendrier 2025.

Économique : Hosting fee estimé (proxy économique).

Médiatique : Audience / affluence estimée (proxy d'impact médiatique).

Climatique : Température moyenne mensuelle (°C) et précipitations mensuelles moyennes (mm).

Note sur les données : Ces informations proviennent de sources publiques (calendrier officiel, bases de données climatiques) et sont complétées, lorsque nécessaire, par des proxys explicitement assumés pour les besoins de la modélisation.

## Méthodologie (résumé)

L'approche repose sur les principes suivants :

Distances et Logistique : Calcul des distances géodésiques entre circuits via la formule de Haversine. Les émissions de CO₂ sont estimées via un proxy proportionnel à la distance parcourue.

Contraintes Temporelles : Modélisation des contraintes logistiques par l'application de marges temporelles minimales entre deux événements.

Critères Économiques et Climatiques : Normalisation des données brutes (températures, coûts, audience) pour les rendre comparables.

Algorithme ELECTRE IS : Application de la méthode de surclassement pour établir des relations de préférence entre les alternatives.

L’ensemble des hypothèses de modélisation est détaillé dans le rapport associé à ce projet.

## Utilisation

Pour exécuter l'analyse, assurez-vous de disposer de Python et des bibliothèques nécessaires (telles que pandas et numpy).

Exécutez le script principal via le terminal :

python electre_is_incremental.py


Le script charge le fichier CSV, procède au calcul des critères, exécute l'algorithme ELECTRE IS et génère le classement ou la sélection incrémentale des Grands Prix.

## Sources principales

Calendrier officiel Formule 1 – Saison 2025

Ergast / Kaggle Formula 1 Dataset (Données géographiques des circuits)

Open-Meteo / NOAA / Meteostat (Historiques climatiques)

Rapports et presse spécialisée (Estimations des hosting fees et audiences)
