-- ============================================================
-- ÉTAPE 1 — Requêtes SQL avancées
-- Projet : Analyse & Prédiction des flux de mobilité IDFM
-- ============================================================

-- -------------------------------------------------------------
-- POURQUOI CES REQUÊTES ?
-- Ces requêtes servent à deux choses :
-- 1. Comprendre les données avant de modéliser (exploration)
-- 2. Créer des features qu'on utilisera dans nos modèles ML
-- Elles utilisent des techniques avancées : window functions,
-- CTEs, agrégations — ce qu'on voit rarement avec juste Pandas.
-- -------------------------------------------------------------

USE idfm_mobility;

-- ============================================================
-- REQUÊTE 1 — TOP 10 arrêts les plus fréquentés
-- ============================================================
-- POURQUOI : savoir quels arrêts concentrent le plus de trafic.
-- Ces arrêts seront les plus importants dans nos modèles.
-- GROUP BY regroupe toutes les lignes d'un même arrêt,
-- SUM additionne leurs validations, ORDER BY trie du plus grand
-- au plus petit, LIMIT garde seulement les 10 premiers.
-- ============================================================
SELECT
    LIBELLE_ARRET,
    SUM(NB_VALD) AS total_validations,
    COUNT(DISTINCT JOUR) AS nb_jours_actifs
FROM validations
GROUP BY LIBELLE_ARRET
ORDER BY total_validations DESC
LIMIT 10;


-- ============================================================
-- REQUÊTE 2 — Trafic moyen par jour de la semaine
-- ============================================================
-- POURQUOI : comprendre les patterns hebdomadaires.
-- Un lundi est-il plus chargé qu'un dimanche ?
-- DAYOFWEEK retourne 1=Dimanche, 2=Lundi ... 7=Samedi en MySQL.
-- CASE WHEN transforme le numéro en nom lisible.
-- AVG calcule la moyenne des validations par jour de semaine.
-- ============================================================
WITH trafic_par_jour AS (
    -- Étape 1 : calculer le trafic total de chaque jour
    SELECT
        JOUR,
        DAYOFWEEK(JOUR) AS num_jour,
        SUM(NB_VALD) AS total_jour
    FROM validations
    GROUP BY JOUR, DAYOFWEEK(JOUR)
)
-- Étape 2 : calculer la moyenne par jour de semaine
SELECT
    num_jour,
    CASE num_jour
        WHEN 1 THEN 'Dimanche'
        WHEN 2 THEN 'Lundi'
        WHEN 3 THEN 'Mardi'
        WHEN 4 THEN 'Mercredi'
        WHEN 5 THEN 'Jeudi'
        WHEN 6 THEN 'Vendredi'
        WHEN 7 THEN 'Samedi'
    END AS nom_jour,
    COUNT(JOUR) AS nb_jours,
    ROUND(AVG(total_jour), 0) AS moyenne_trafic_journalier,
    SUM(total_jour) AS total_validations
FROM trafic_par_jour
GROUP BY num_jour
ORDER BY num_jour;


-- ============================================================
-- REQUÊTE 3 — Window Function : variation semaine sur semaine
-- ============================================================
-- POURQUOI : c'est exactement notre feature "lag7" pour XGBoost.
-- LAG(colonne, N) est une window function qui récupère la valeur
-- de la ligne N positions avant dans la fenêtre.
-- PARTITION BY LIBELLE_ARRET = on fait ça séparément pour chaque arrêt
-- ORDER BY JOUR = on ordonne par date avant d'appliquer le lag
-- Résultat : pour chaque arrêt et chaque jour, on voit le trafic
-- de ce jour ET celui du même arrêt 7 jours avant.
-- ============================================================
SELECT
    LIBELLE_ARRET,
    JOUR,
    SUM(NB_VALD) AS nb_vald_jour,

    -- Trafic du même arrêt il y a 7 jours (notre future feature lag7)
    LAG(SUM(NB_VALD), 7) OVER (
        PARTITION BY LIBELLE_ARRET
        ORDER BY JOUR
    ) AS nb_vald_lag7,

    -- Variation en % par rapport à il y a 7 jours
    ROUND(
        (SUM(NB_VALD) - LAG(SUM(NB_VALD), 7) OVER (
            PARTITION BY LIBELLE_ARRET ORDER BY JOUR
        )) / LAG(SUM(NB_VALD), 7) OVER (
            PARTITION BY LIBELLE_ARRET ORDER BY JOUR
        ) * 100,
    1) AS variation_pct_7j

FROM validations
GROUP BY LIBELLE_ARRET, JOUR
ORDER BY LIBELLE_ARRET, JOUR;


-- ============================================================
-- REQUÊTE 4 — CTE : score de congestion cumulé par arrêt
-- ============================================================
-- POURQUOI : le score de congestion mesure à quel point un arrêt
-- est proche de sa capacité maximale. C'est une feature utile
-- pour le dashboard Power BI et pour le clustering.
-- UNE CTE (Common Table Expression) = un résultat intermédiaire
-- qu'on nomme et qu'on peut réutiliser dans la même requête.
-- C'est comme créer une variable temporaire en Python.
-- ============================================================
WITH trafic_par_arret AS (
    -- Étape 1 : calculer le trafic total par arrêt et par jour
    SELECT
        LIBELLE_ARRET,
        JOUR,
        SUM(NB_VALD) AS nb_vald_jour
    FROM validations
    GROUP BY LIBELLE_ARRET, JOUR
),
max_par_arret AS (
    -- Étape 2 : trouver le maximum historique de chaque arrêt
    SELECT
        LIBELLE_ARRET,
        MAX(nb_vald_jour) AS max_vald
    FROM trafic_par_arret
    GROUP BY LIBELLE_ARRET
)
-- Étape 3 : calculer le score = trafic du jour / maximum historique
-- Un score de 1.0 = l'arrêt est à son maximum historique
-- Un score de 0.3 = l'arrêt est à 30% de sa capacité maximale
SELECT
    t.LIBELLE_ARRET,
    t.JOUR,
    t.nb_vald_jour,
    m.max_vald,
    ROUND(t.nb_vald_jour / m.max_vald, 3) AS score_congestion
FROM trafic_par_arret t
JOIN max_par_arret m ON t.LIBELLE_ARRET = m.LIBELLE_ARRET
ORDER BY score_congestion DESC
LIMIT 20;


-- ============================================================
-- REQUÊTE 5 — Taux de titres NON DEFINIS par arrêt
-- ============================================================
-- POURQUOI : les lignes CATEGORIE_TITRE = "NON DEFINI" peuvent
-- signaler de la fraude ou des problèmes de lecture de titre.
-- Les arrêts avec >20% de NON DEFINIS sont suspects.
-- On utilise une expression conditionnelle dans SUM pour compter
-- seulement les lignes NON DEFINI (c'est du SQL conditionnel).
-- ============================================================
SELECT
    LIBELLE_ARRET,
    SUM(NB_VALD) AS total_validations,

    -- Compter uniquement les validations NON DEFINI
    SUM(CASE WHEN CATEGORIE_TITRE = 'NON DEFINI' THEN NB_VALD ELSE 0 END) AS nb_non_defini,

    -- Calculer le taux en pourcentage
    ROUND(
        SUM(CASE WHEN CATEGORIE_TITRE = 'NON DEFINI' THEN NB_VALD ELSE 0 END)
        / SUM(NB_VALD) * 100,
    1) AS taux_non_defini_pct

FROM validations
GROUP BY LIBELLE_ARRET

-- Garder uniquement les arrêts avec plus de 20% de NON DEFINIS


ORDER BY taux_non_defini_pct DESC
limit 20;